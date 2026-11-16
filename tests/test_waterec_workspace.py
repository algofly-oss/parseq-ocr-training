import pytest
from parseq_ocr.waterec_workspace import ScratchPool


class Allocation:
    def __init__(self, address):
        self.address = address
        self.frees = 0

    def __int__(self):
        return self.address

    def free(self):
        self.frees += 1


def test_sequential_context_lifetime_and_independent_recognizers():
    first, second = Allocation(256), Allocation(512)
    sizes = []
    a = ScratchPool(lambda n: (sizes.append(n), first)[1])
    b = ScratchPool(lambda n: second)
    contexts = [a.acquire(n, 11) for n in [4000, 1600, 1000, 1700]]
    other = b.acquire(4000, 22)
    assert sizes == [4000]
    assert {int(x) for x in contexts} == {256}
    assert int(other) == 512
    for i in [2, 3, 1]:
        contexts[i].free()
        contexts[i].free()
        assert first.frees == 0
        assert int(contexts[0]) == 256
    contexts[0].free()
    assert first.frees == 1 and a.closed and a.active_leases == 0
    assert int(other) == 512 and second.frees == 0
    other.free()
    assert second.frees == 1
    with pytest.raises(RuntimeError, match="closed"):
        int(contexts[0])
    with pytest.raises(RuntimeError, match="closed"):
        a.acquire(1, 11)


def test_reject_growth_cross_stream_and_premature_release():
    buffer = Allocation(256)
    pool = ScratchPool(lambda n: buffer)
    lease = pool.acquire(100, 11)
    with pytest.raises(ValueError, match="same CUDA stream"):
        pool.acquire(50, 22)
    with pytest.raises(ValueError, match="cannot grow"):
        pool.acquire(101, 11)
    with pytest.raises(RuntimeError, match="execution contexts"):
        pool.close()
    assert pool.active_leases == 1 and int(lease) == 256 and buffer.frees == 0
    lease.free()
    pool.close()
    assert buffer.frees == 1


def test_failed_backend_free_retains_ownership_for_retry():
    class FailingAllocation(Allocation):
        def free(self):
            self.frees += 1
            if self.frees == 1:
                raise RuntimeError("backend failure")
    buffer = FailingAllocation(256)
    pool = ScratchPool(lambda n: buffer)
    lease = pool.acquire(0, 11)
    assert pool.capacity == 1
    with pytest.raises(RuntimeError, match="backend failure"):
        lease.free()
    assert pool.active_leases == 1 and int(lease) == 256
    lease.free()
    assert pool.closed and pool.active_leases == 0 and buffer.frees == 2
