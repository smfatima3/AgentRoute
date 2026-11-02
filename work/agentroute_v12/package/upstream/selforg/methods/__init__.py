from .mas_base import MAS
from .selforg import SelfOrg_Main

method2class = {
    "vanilla": MAS,
    "selforg": SelfOrg_Main,
}


def get_method_class(method_name, dataset_name=None):
    method_name = method_name.lower()
    all_method_names = method2class.keys()
    matched_method_names = [
        sample_method_name
        for sample_method_name in all_method_names
        if method_name in sample_method_name
    ]

    if len(matched_method_names) > 0:
        method_name = matched_method_names[0]
    else:
        raise ValueError(
            f"[ERROR] No method found matching {method_name}. Please check the method name."
        )

    return method2class[method_name]
