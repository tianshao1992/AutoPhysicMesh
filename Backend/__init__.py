

# References: https://github.com/dmlc/dgl/tree/master/python/dgl/backend

import importlib
import json
import os
import sys

from mpo_engine.backend import backend
from mpo_engine.backend.set_default_backend import set_default_backend
from mpo_engine.backend.utils import get_available_backend, interactive_install_paddle, verify_backend

_enabled_apis = set()

def is_enabled(api):
    """Return true if the api is enabled by the current backend.

    Args:
        api (string): The api name.

    Returns:
        bool: ``True`` if the API is enabled by the current backend.
    """
    return api in _enabled_apis


def _gen_missing_api(api, mod_name):
    """Generate a missing API function."""
    def _missing_api(*args, **kwargs):
        """
        Raise ImportError if the API is not available.
        """
        raise ImportError(
            'API "%s" is not supported by backend "%s".'
            " You can switch to other backends by setting"
            " the MPO_BACKEND environment." % (api, mod_name)
        )

    return _missing_api


def backend_message(backend_name):
    """Show message about backend.

    Args:
        backend_name: which backend used
    """
    msg = f"Using backend: {backend_name}\nOther supported backends: "
    if backend_name == "taichi":
        msg += "pytorch, jax, paddle.\n"
    elif backend_name == "pytorch":
        msg += "taichi, jax, paddle.\n"
    elif backend_name == "jax":
        msg += "taichi, pytorch, paddle.\n"
    elif backend_name == "paddle":
        msg += "taichi, pytorch, jax.\n"
    print(msg, file=sys.stderr, flush=True)


def _load_mod(mod_name, base_name):
    """
    Load the module by name.
    """
    # load backend module
    mod = importlib.import_module(".%s" % mod_name, base_name)
    # mod = importlib.import_module(mod_name)
    thismod = sys.modules[base_name]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


def load_backend(mod_name):
    """Load backend module."""
    backend_message(mod_name)
    # print(".%s" % mod_name.replace(".", "_"), __name__)
    mod = importlib.import_module(".%s" % mod_name.replace(".", "_"), __name__)
    # mod = importlib.import_module(mod_name)
    thismod = sys.modules[__name__]
    # log backend name
    setattr(thismod, "backend_name", mod_name)
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for backend "%s".'
                    % mod_name
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, "data_type_dict", data_type_dict)
            setattr(
                thismod,
                "reverse_data_type_dict",
                {v: k for k, v in data_type_dict.items()},
            )
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))


def get_preferred_backend():
    """
    Get preferred backend.
    """
    backend_name = None
    # User-selected backend
    config_path = os.path.join(os.path.expanduser("~"), ".mpo", "config.json")
    if "MPO_BACKEND" in os.environ:
        backend_name = os.getenv("MPO_BACKEND")
    # Backward compatibility
    elif "MPO_BACKEND" in os.environ:
        backend_name = os.getenv("MPO_BACKEND")
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get("backend", "").lower()
    if backend_name is not None:
        verify_backend(backend_name)
        return backend_name
    # No backend selected
    print("No backend selected.")

    # Find available backend
    print("Finding available backend...")
    backend_name = get_available_backend()
    if backend_name is not None:
        print(f"Found {backend_name}")
        set_default_backend(backend_name)
        return backend_name

    # No backend available
    print("Cannot find available backend.")
    interactive_install_paddle()
    set_default_backend("paddle")
    return "paddle"

backend_name = get_preferred_backend()
# print(backend_name)
load_backend(backend_name)

