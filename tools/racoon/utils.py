import os, torch

# Xác định xem có Just-in-time compile hay không?
# Phụ thuộc vào bên trong hàm có dùng tới gradient checkpoint hay cpu off-load hay không?
if os.environ.get("RWKV_JIT_ON", "1") == "1":
    # Enable JIT
    JITableModule = torch.jit.ScriptModule
    JITableFunction = torch.jit.script_method
else:
    # Disable JIT
    def __nop(obj): return obj
    JITableModule = torch.nn.Module
    JITableFunction = __nop
