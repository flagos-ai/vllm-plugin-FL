"""Per-worker capacity reservation, applied after an unreserved native profile.

This is installed on one explicit worker instance, never a global class. Auxiliary
buffers must be allocated only after reservation, so they are not counted twice
by the native profile. Neither model/profile execution nor graph policy changes.
"""
import types
from dataclasses import dataclass


@dataclass(frozen=True)
class AuxReservationReceipt:
    native_unreserved_available: int
    requested_aux_bytes: int
    reserved_aux_bytes: int
    page_allocation_rounding_headroom: int
    cache_available: int
    allocation_rounding_quantum: int


def rounded_reservation_bytes(budget, quantum=512):
    if type(quantum) is not int or quantum < 1 or quantum & (quantum-1):
        raise ValueError('allocation rounding quantum must be a power of two')
    return sum((size+quantum-1)//quantum*quantum for size in budget.byte_breakdown().values())


def install_aux_reservation(worker, budget, *, allocation_rounding_quantum=512):
    if getattr(worker, '_c500_aux_budget', None) is not None:
        raise RuntimeError('mixed auxiliary reservation is already installed on this worker')
    if getattr(worker, '_c500_aux_allocation', None) is not None:
        raise RuntimeError('auxiliary buffers already exist and may have been included in native profiling')
    reserve = rounded_reservation_bytes(budget, allocation_rounding_quantum)
    page_rounding = budget.layers*(allocation_rounding_quantum-1)
    original = worker.determine_available_memory
    worker._c500_aux_budget = budget
    worker._c500_aux_reservation = None

    def determine(this):
        if getattr(this, '_c500_aux_allocation', None) is not None:
            raise RuntimeError('reprofiling with allocated mixed auxiliary buffers would invalidate reservation accounting')
        available = original()
        if type(available) is not int or available <= reserve+page_rounding:
            raise ValueError('native available memory cannot cover mixed auxiliary reservation')
        receipt = AuxReservationReceipt(available, budget.total_bytes, reserve, page_rounding,
                                        available-reserve-page_rounding, allocation_rounding_quantum)
        this._c500_aux_reservation = receipt
        return receipt.cache_available

    worker.determine_available_memory = types.MethodType(determine, worker)
    return original


def verify_before_allocation(worker, kv_cache_config):
    receipt = getattr(worker, '_c500_aux_reservation', None)
    if receipt is None:
        raise RuntimeError('native profile and mixed auxiliary reservation must run before cache allocation')
    page_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    # Include the same explicit allocator rounding for backing tensors. An
    # override that exceeds the actual profiled budget is rejected before OOM.
    q = receipt.allocation_rounding_quantum
    rounded_pages = sum((t.size+q-1)//q*q for t in kv_cache_config.kv_cache_tensors)
    if rounded_pages > receipt.native_unreserved_available-receipt.reserved_aux_bytes:
        raise ValueError('framework cache capacity or explicit block override exceeds reserved byte budget')
    return {'page_requested_bytes': page_bytes, 'page_rounded_bytes': rounded_pages,
            'aux_requested_bytes': receipt.requested_aux_bytes,
            'aux_reserved_bytes': receipt.reserved_aux_bytes,
            'page_rounding_headroom': receipt.page_allocation_rounding_headroom,
            'native_unreserved_available': receipt.native_unreserved_available,
            'remaining_bytes': receipt.native_unreserved_available-rounded_pages-receipt.reserved_aux_bytes}
