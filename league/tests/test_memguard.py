from league import memguard

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                3560.
Pages active:                            169930.
Pages inactive:                          161831.
Pages speculative:                         7450.
Pages throttled:                              0.
Pages wired down:                        180911.
"""


def test_parse_vm_stat_free_inactive_speculative():
    assert memguard.available_bytes(VM_STAT) == (3560 + 161831 + 7450) * 16384


def test_enough_memory_threshold(monkeypatch):
    monkeypatch.setattr(memguard, "available_bytes", lambda text=None: 3 * 2**30 - 1)
    assert not memguard.enough_memory()
    monkeypatch.setattr(memguard, "available_bytes", lambda text=None: 3 * 2**30)
    assert memguard.enough_memory()
