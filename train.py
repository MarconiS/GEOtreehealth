import tree_health_detection
# reload the module
import importlib
import config
importlib.reload(config)
importlib.reload(tree_health_detection)

tree_health_detection.__main__()