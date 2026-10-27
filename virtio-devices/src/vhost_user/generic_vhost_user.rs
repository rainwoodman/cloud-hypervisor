// Copyright 2019 Intel Corporation. All Rights Reserved.
// Copyright 2025 Demi Marie Obenour.
// SPDX-License-Identifier: Apache-2.0

use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::{self, JoinHandle};
use std::{io, result};

use event_monitor::event;
use log::{error, info, warn};
use seccompiler::{SeccompAction, apply_filter};
use vhost::vhost_user::message::{
    VhostUserConfigFlags, VhostUserMMap, VhostUserMMapFlags, VhostUserProtocolFeatures,
    VhostUserVirtioFeatures,
};
use vhost::vhost_user::{FrontendReqHandler, VhostUserFrontend, VhostUserFrontendReqHandler};
use vm_device::UserspaceMapping;
use vm_migration::protocol::MemoryRangeTable;
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::EventFd;

use super::vu_common_ctrl::VhostUserHandle;
use super::{Error, Result};
use crate::device::ActivationContext;
use crate::seccomp_filters::{Thread, get_seccomp_filter};
use crate::vhost_user::{VhostUserCommon, VhostUserState};
use crate::{
    ActivateResult, GuestRegionMmap, VIRTIO_F_ACCESS_PLATFORM, VirtioCommon, VirtioDevice,
    VirtioInterrupt, VirtioInterruptType, VirtioSharedMemoryList,
};

pub type State = VhostUserState<()>;

struct BackendReqHandler {
    interrupt_cb: Mutex<Option<Arc<dyn VirtioInterrupt>>>,
    shm_regions: Arc<Mutex<Option<VirtioSharedMemoryList>>>,
}

impl BackendReqHandler {
    fn region_target_addr(&self, req: &VhostUserMMap) -> io::Result<*mut libc::c_void> {
        let guard = self.shm_regions.lock().unwrap();
        let shm_list = guard
            .as_ref()
            .ok_or_else(|| io::Error::other("No shared memory region configured"))?;
        let region = shm_list
            .region_list
            .get(usize::from(req.shmid))
            .filter(|r| r.len > 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid shmid"))?;
        if req
            .shm_offset
            .checked_add(req.len)
            .is_none_or(|end| end > region.len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared memory mapping out of bounds",
            ));
        }
        let offset = (region.offset + req.shm_offset) as usize;
        Ok(shm_list.mapping.as_ptr().wrapping_add(offset).cast())
    }
}

impl VhostUserFrontendReqHandler for BackendReqHandler {
    fn handle_config_change(&self) -> io::Result<u64> {
        if let Some(ref interrupt_cb) = *self.interrupt_cb.lock().unwrap() {
            interrupt_cb
                .trigger(VirtioInterruptType::Config)
                .map_err(|e| {
                    error!("Failed to signal config change: {e:?}");
                    io::Error::other(e)
                })?;
        }
        Ok(0)
    }

    fn shmem_map(&self, req: &VhostUserMMap, fd: &dyn AsRawFd) -> io::Result<u64> {
        let target_addr = self.region_target_addr(req)?;
        let mut prot = libc::PROT_READ;
        if req.flags & VhostUserMMapFlags::WRITABLE.bits() != 0 {
            prot |= libc::PROT_WRITE;
        }
        // SAFETY: `target_addr` and `req.len` have been bounds-checked against
        // the `MmapRegion` backing the shared memory BAR, and `MAP_FIXED`
        // atomically replaces only that sub-range.
        let ret = unsafe {
            libc::mmap(
                target_addr,
                req.len as usize,
                prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd.as_raw_fd(),
                req.fd_offset as libc::off_t,
            )
        };
        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(0)
    }

    fn shmem_unmap(&self, req: &VhostUserMMap) -> io::Result<u64> {
        let target_addr = self.region_target_addr(req)?;
        // SAFETY: `target_addr` and `req.len` have been bounds-checked against
        // the `MmapRegion` backing the shared memory BAR, and `MAP_FIXED`
        // atomically restores that sub-range to an unmapped `PROT_NONE` guard.
        let ret = unsafe {
            libc::mmap(
                target_addr,
                req.len as usize,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(0)
    }
}

pub struct GenericVhostUser {
    vu_common: VhostUserCommon,
    id: String,
    // Hold ownership of the memory that is allocated for the device
    // which will be automatically dropped when the device is dropped
    cache: Arc<Mutex<Option<VirtioSharedMemoryList>>>,
    shmem_sizes: Vec<u64>,
    backend_req_handler: Arc<BackendReqHandler>,
    pre_activate_worker: Option<(EventFd, JoinHandle<FrontendReqHandler<BackendReqHandler>>)>,
    seccomp_action: SeccompAction,
    exit_evt: EventFd,
    cfg_warning: AtomicBool,
}

impl GenericVhostUser {
    /// Create a new generic vhost-user device.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        path: &str,
        request_queue_sizes: Vec<u16>,
        device_type: u32,
        cache: Option<VirtioSharedMemoryList>,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        access_platform_enabled: bool,
        state: Option<State>,
    ) -> Result<GenericVhostUser> {
        // Calculate the actual number of queues needed.
        let num_queues =
            u16::try_from(request_queue_sizes.len()).map_err(|_| Error::BadQueueNum)?;

        // Connect to the vhost-user socket.
        let mut vu =
            VhostUserHandle::connect_vhost_user(false, path, num_queues, false, &exit_evt, |_| {
                Ok(())
            })?;

        let cache = Arc::new(Mutex::new(cache));
        let backend_req_handler = Arc::new(BackendReqHandler {
            interrupt_cb: Mutex::new(None),
            shm_regions: Arc::clone(&cache),
        });

        let (
            avail_features,
            acked_features,
            acked_protocol_features,
            vu_num_queues,
            paused,
            vring_bases,
            shmem_sizes,
            pre_activate_worker,
        ) = if let Some(state) = state {
            info!("Restoring generic vhost-user {id}");
            vu.set_protocol_features_vhost_user(
                state.acked_features,
                state.acked_protocol_features,
            )?;

            vu.restore_state(&state)?;

            (
                state.avail_features,
                state.acked_features,
                state.acked_protocol_features,
                state.vu_num_queues,
                true,
                state.vring_bases,
                Vec::new(),
                None,
            )
        } else {
            let avail_protocol_features = VhostUserProtocolFeatures::CONFIG
                | VhostUserProtocolFeatures::MQ
                | VhostUserProtocolFeatures::CONFIGURE_MEM_SLOTS
                | VhostUserProtocolFeatures::REPLY_ACK
                | VhostUserProtocolFeatures::INFLIGHT_SHMFD
                | VhostUserProtocolFeatures::LOG_SHMFD
                | VhostUserProtocolFeatures::DEVICE_STATE
                | VhostUserProtocolFeatures::BACKEND_REQ
                | VhostUserProtocolFeatures::SHMEM;

            let avail_features = super::DEFAULT_VIRTIO_FEATURES;

            let (acked_features, acked_protocol_features) =
                vu.negotiate_features_vhost_user(avail_features, avail_protocol_features)?;

            let backend_num_queues =
                if acked_protocol_features & VhostUserProtocolFeatures::MQ.bits() != 0 {
                    vu.socket_handle()
                        .get_queue_num()
                        .map_err(Error::VhostUserGetQueueMaxNum)?
                } else {
                    num_queues as u64
                };

            if num_queues as u64 > backend_num_queues {
                error!(
                    "generic vhost-user requested too many queues ({num_queues}) \
since the backend only supports {backend_num_queues}\n",
                );
                return Err(Error::BadQueueNum);
            }

            let shmem_sizes =
                if acked_protocol_features & VhostUserProtocolFeatures::SHMEM.bits() != 0 {
                    let cfg = vu
                        .socket_handle()
                        .get_shmem_config()
                        .map_err(Error::VhostUserGetShmemConfig)?;
                    cfg.memory_sizes[..(cfg.nregions as usize)].to_vec()
                } else {
                    Vec::new()
                };

            let pre_activate_worker = if acked_protocol_features
                & VhostUserProtocolFeatures::BACKEND_REQ.bits()
                != 0
            {
                let mut handler = FrontendReqHandler::new(Arc::clone(&backend_req_handler))
                    .map_err(Error::FrontendReqHandlerCreation)?;

                if acked_protocol_features & VhostUserProtocolFeatures::REPLY_ACK.bits() != 0 {
                    handler.set_reply_ack_flag(true);
                }

                vu.socket_handle()
                    .set_backend_request_fd(&handler.get_tx_raw_fd())
                    .map_err(Error::VhostUserSetBackendRequestFd)?;

                let kill_evt = EventFd::new(libc::EFD_NONBLOCK).map_err(Error::CreateEventFd)?;
                let epoll = Epoll::new().map_err(Error::EpollCreate)?;
                epoll
                    .ctl(
                        ControlOperation::Add,
                        handler.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, 0),
                    )
                    .map_err(Error::EpollCtl)?;
                epoll
                    .ctl(
                        ControlOperation::Add,
                        kill_evt.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, 1),
                    )
                    .map_err(Error::EpollCtl)?;

                let seccomp_filter =
                    get_seccomp_filter(&seccomp_action, Thread::VirtioGenericVhostUser)
                        .map_err(Error::CreateSeccompFilter)?;
                let thread_exit_evt = exit_evt.try_clone().map_err(Error::CreateEventFd)?;
                let handle = thread::Builder::new()
                    .name(format!("{id}_backend_req"))
                    .spawn(move || {
                        if !seccomp_filter.is_empty()
                            && let Err(e) = apply_filter(&seccomp_filter)
                        {
                            error!("Error applying seccomp filter: {e:?}");
                            thread_exit_evt.write(1).ok();
                            return handler;
                        }
                        let mut events = [EpollEvent::default(); 2];
                        loop {
                            let num_events = match epoll.wait(-1, &mut events) {
                                Ok(n) => n,
                                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                                Err(e) => {
                                    error!("BackendReqHandler epoll wait error: {e:?}");
                                    break;
                                }
                            };
                            let mut stop = false;
                            for event in events.iter().take(num_events) {
                                if event.data() == 1 {
                                    stop = true;
                                    break;
                                }
                                if event.event_set().intersects(
                                    EventSet::ERROR | EventSet::HANG_UP | EventSet::READ_HANG_UP,
                                ) {
                                    stop = true;
                                    break;
                                }
                                if event.event_set().contains(EventSet::IN)
                                    && let Err(e) = handler.handle_request()
                                {
                                    error!("BackendReqHandler error: {e:?}");
                                    stop = true;
                                    break;
                                }
                            }
                            if stop {
                                break;
                            }
                        }
                        handler
                    })
                    .map_err(Error::ThreadSpawn)?;
                Some((kill_evt, handle))
            } else {
                None
            };

            let mut avail_features = acked_features;
            if access_platform_enabled {
                avail_features |= 1u64 << VIRTIO_F_ACCESS_PLATFORM;
            }

            (
                avail_features,
                // If part of the available features that have been acked, the
                // PROTOCOL_FEATURES bit must be already set through the VIRTIO
                // acked features as we know the guest would never ack it, thus
                // the feature would be lost.
                acked_features & VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits(),
                acked_protocol_features,
                num_queues,
                false,
                None,
                shmem_sizes,
                pre_activate_worker,
            )
        };

        Ok(GenericVhostUser {
            vu_common: VhostUserCommon {
                virtio_common: VirtioCommon {
                    device_type,
                    avail_features,
                    acked_features,
                    queue_sizes: request_queue_sizes,
                    paused_sync: Some(Arc::new(Barrier::new(2))),
                    min_queues: 1,
                    paused: Arc::new(AtomicBool::new(paused)),
                    ..Default::default()
                },
                vu: Some(Arc::new(Mutex::new(vu))),
                acked_protocol_features,
                socket_path: path.to_string(),
                vu_num_queues,
                vring_bases,
                ..Default::default()
            },
            id,
            cache,
            shmem_sizes,
            backend_req_handler,
            pre_activate_worker,
            seccomp_action,
            exit_evt,
            cfg_warning: AtomicBool::new(false),
        })
    }

    pub fn negotiated_shmem_sizes(&self) -> &[u64] {
        &self.shmem_sizes
    }

    fn stop_pre_activate_worker(&mut self) -> Option<FrontendReqHandler<BackendReqHandler>> {
        if let Some((kill_evt, handle)) = self.pre_activate_worker.take() {
            let _ = kill_evt.write(1);
            handle.join().ok()
        } else {
            None
        }
    }

    fn state(&self) -> result::Result<State, MigratableError> {
        self.vu_common.state(())
    }

    #[cold]
    #[inline(never)]
    fn warn_no_config_access(&self) {
        if self
            .cfg_warning
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            warn!(
                "Attempt to read config space, but backend does not support config \
space access. Reads will return 0xFF and writes will be ignored."
            );
        }
    }
}

impl Drop for GenericVhostUser {
    fn drop(&mut self) {
        let _ = self.stop_pre_activate_worker();
        self.vu_common.shutdown();
    }
}

impl VirtioDevice for GenericVhostUser {
    fn device_type(&self) -> u32 {
        self.vu_common.virtio_common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.vu_common.virtio_common.queue_sizes
    }

    fn features(&self) -> u64 {
        self.vu_common.virtio_common.avail_features
    }

    fn ack_features(&mut self, value: u64) {
        self.vu_common.virtio_common.ack_features(value);
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        if (VhostUserProtocolFeatures::CONFIG.bits() & self.vu_common.acked_protocol_features) == 0
        {
            self.warn_no_config_access();

            data.fill(0xFF);
            return;
        }
        if let Err(e) = self
            .vu_common
            .vu
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .socket_handle()
            .get_config(
                offset.try_into().unwrap(),
                data.len().try_into().unwrap(),
                VhostUserConfigFlags::empty(),
                data,
            )
            .map(|(_, config)| data.copy_from_slice(&config))
        {
            panic!("Failed getting generic vhost-user configuration: {e}");
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        if (VhostUserProtocolFeatures::CONFIG.bits() & self.vu_common.acked_protocol_features) == 0
        {
            self.warn_no_config_access();
            return;
        }
        if let Err(e) = self
            .vu_common
            .vu
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .socket_handle()
            .set_config(
                offset.try_into().unwrap(),
                VhostUserConfigFlags::WRITABLE,
                data,
            )
        {
            panic!("Failed setting generic vhost-user configuration: {e}");
        }
    }

    fn activate(&mut self, context: ActivationContext) -> ActivateResult {
        let ActivationContext {
            mem,
            interrupt_cb,
            queues,
            device_status,
        } = context;
        self.vu_common
            .virtio_common
            .activate(&queues, Arc::clone(&interrupt_cb))?;
        *self.backend_req_handler.interrupt_cb.lock().unwrap() = Some(Arc::clone(&interrupt_cb));

        let has_backend_req = self.vu_common.acked_protocol_features
            & VhostUserProtocolFeatures::BACKEND_REQ.bits()
            != 0;

        let backend_req_handler = if let Some(handler) = self.stop_pre_activate_worker() {
            Some(handler)
        } else {
            has_backend_req
                .then(|| {
                    let mut handler = FrontendReqHandler::new(Arc::clone(
                        &self.backend_req_handler,
                    ))
                    .map_err(|e| {
                        crate::ActivateError::VhostUserSetup(Error::FrontendReqHandlerCreation(e))
                    })?;

                    if self.vu_common.acked_protocol_features
                        & VhostUserProtocolFeatures::REPLY_ACK.bits()
                        != 0
                    {
                        handler.set_reply_ack_flag(true);
                    }

                    Ok(handler)
                })
                // Return inner Err early, keep Option of `Ok` value
                .transpose()?
        };

        // Run a dedicated thread for handling potential reconnections with
        // the backend.
        let (kill_evt, pause_evt) = self.vu_common.virtio_common.dup_eventfds()?;

        let mut handler = self.vu_common.activate(
            mem,
            &queues,
            Arc::clone(&interrupt_cb),
            self.vu_common.virtio_common.acked_features,
            backend_req_handler,
            kill_evt,
            pause_evt,
        )?;

        let paused = Arc::clone(&self.vu_common.virtio_common.paused);
        let paused_sync = self.vu_common.virtio_common.paused_sync.clone();

        self.vu_common.spawn_worker(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioGenericVhostUser,
            &self.exit_evt,
            Arc::clone(&device_status),
            Arc::clone(&interrupt_cb),
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        )?;

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) {
        self.vu_common.reset(&self.id);
    }

    fn shutdown(&mut self) {
        let _ = self.stop_pre_activate_worker();
        self.vu_common.shutdown();
    }

    fn get_shm_regions(&self) -> Option<VirtioSharedMemoryList> {
        self.cache.lock().unwrap().clone()
    }

    fn set_shm_regions(
        &mut self,
        shm_regions: VirtioSharedMemoryList,
    ) -> result::Result<(), crate::Error> {
        *self.cache.lock().unwrap() = Some(shm_regions);
        Ok(())
    }

    fn add_memory_region(
        &mut self,
        region: &Arc<GuestRegionMmap>,
    ) -> result::Result<(), crate::Error> {
        self.vu_common.add_memory_region(region)
    }

    fn userspace_mappings(&self) -> Vec<UserspaceMapping> {
        let mut mappings = Vec::new();
        if let Some(cache) = self.cache.lock().unwrap().as_ref() {
            mappings.push(UserspaceMapping {
                mem_slot: cache.mem_slot,
                addr: cache.addr,
                mapping: Arc::clone(&cache.mapping),
                mergeable: false,
            });
        }

        mappings
    }
}

impl Pausable for GenericVhostUser {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.pause()?;
        self.vu_common.virtio_common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.virtio_common.resume()?;
        self.vu_common.resume()
    }
}

impl Snapshottable for GenericVhostUser {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        self.vu_common.snapshot(&self.state()?)
    }
}
impl Transportable for GenericVhostUser {}

impl Migratable for GenericVhostUser {
    fn start_dirty_log(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.start_dirty_log()
    }

    fn stop_dirty_log(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.stop_dirty_log()
    }

    fn dirty_log(&mut self) -> result::Result<MemoryRangeTable, MigratableError> {
        self.vu_common.dirty_log()
    }

    fn notify_started_migration(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.notify_started_migration()
    }

    fn notify_failed_migration(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.notify_failed_migration()
    }

    fn notify_completed_migration(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.notify_completed_migration()
    }
}
