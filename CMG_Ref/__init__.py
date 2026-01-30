"""
CMG Reference Motion Generator Package

This package provides the Conditional Motion Generator (CMG) for generating
reference motions based on velocity commands.
"""

# 延迟导入以避免复杂的包依赖
def _get_cmg_class():
    """动态加载CMG类"""
    import importlib.util
    from pathlib import Path
    
    module_path = str(Path(__file__).parent / "module" / "cmg.py")
    spec = importlib.util.spec_from_file_location("cmg", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载CMG模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CMG

def _get_cmg_trainer_class():
    """动态加载CMGTrainer类"""
    import importlib.util
    from pathlib import Path
    
    module_path = str(Path(__file__).parent / "cmg_trainer.py")
    spec = importlib.util.spec_from_file_location("cmg_trainer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载CMGTrainer模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CMGTrainer

# 创建延迟属性
class _LazyImport:
    def __getattr__(self, name):
        if name == 'CMG':
            return _get_cmg_class()
        elif name == 'CMGTrainer':
            return _get_cmg_trainer_class()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 设置模块级别的属性访问
import sys
_current_module = sys.modules[__name__]
sys.modules[__name__] = _LazyImport()

__all__ = ['CMG', 'CMGTrainer']
