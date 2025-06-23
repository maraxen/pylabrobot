from typing import List, Optional, cast, Any, Callable, Dict, Set
import inspect

from pylabrobot.machines.machine import Machine, need_setup_finished
from pylabrobot.plate_reading.backend import PlateReaderBackend
from pylabrobot.resources import Coordinate, Plate, Resource
from pylabrobot.resources.resource_holder import ResourceHolder


class NoPlateError(Exception):
  pass


class PlateReader(ResourceHolder, Machine):
  """The front end for plate readers. Plate readers are devices that can read luminescence,
  absorbance, or fluorescence from a plate.

  Plate readers are asynchronous, meaning that their methods will return immediately and
  will not block.

  Here's an example of how to use this class in a Jupyter Notebook:

  >>> from pylabrobot.plate_reading.clario_star import CLARIOStar
  >>> pr = PlateReader(backend=CLARIOStar())
  >>> pr.setup()
  >>> await pr.read_luminescence()
  [[value1, value2, value3, ...], [value1, value2, value3, ...], ...
  """

  def __init__(
    self,
    name: str,
    size_x: float,
    size_y: float,
    size_z: float,
    backend: PlateReaderBackend,
    category: Optional[str] = None,
    model: Optional[str] = None,
  ) -> None:
    ResourceHolder.__init__(
      self,
      name=name,
      size_x=size_x,
      size_y=size_y,
      size_z=size_z,
      category=category,
      model=model,
    )
    Machine.__init__(self, backend=backend)
    self.backend: PlateReaderBackend = backend  # fix type

  def assign_child_resource(
    self,
    resource: Resource,
    location: Optional[Coordinate] = None,
    reassign: bool = True,
  ):
    if len(self.children) >= 1:
      raise ValueError("There already is a plate in the plate reader.")
    if not isinstance(resource, Plate):
      raise ValueError("The resource must be a Plate.")

    super().assign_child_resource(resource, location=location, reassign=reassign)

  def get_plate(self) -> Plate:
    if len(self.children) == 0:
      raise NoPlateError("There is no plate in the plate reader.")
    return cast(Plate, self.children[0])

  async def open(self) -> None:
    await self.backend.open()

  async def close(self) -> None:
    await self.backend.close()

  def _check_args(
    self,
    method: Callable,
    backend_kwargs: Dict[str, Any],
    default: Set[str],
  ) -> Set[str]:
    """Checks that the arguments to `method` are valid.

    Args:
      method: Method to check.
      backend_kwargs: Keyword arguments to `method`.

    Raises:
      TypeError: If the arguments are invalid.

    Returns:
      The set of arguments that need to be removed from `backend_kwargs` before passing to `method`.
    """

    default_args = default.union({"self"})

    sig = inspect.signature(method)
    args = {arg: param for arg, param in sig.parameters.items() if arg not in default_args}
    vars_keyword = {
      arg
      for arg, param in sig.parameters.items() if param.kind == inspect.Parameter.VAR_KEYWORD
    }
    args = {
      arg: param
      for arg, param in args.items() if param.kind
      not in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
      }
    }
    non_default = {arg for arg, param in args.items() if param.default == inspect.Parameter.empty}

    backend_kws = set(backend_kwargs.keys())

    missing = non_default - backend_kws
    if len(missing) > 0:
      raise TypeError(f"Missing required arguments: {missing}")

    if len(vars_keyword) > 0:
      return set()

    extra = backend_kws - set(args.keys())
    if len(extra) > 0 and len(vars_keyword) == 0:
      raise TypeError(f"Got unexpected keyword arguments: {extra}")

    return extra

  @need_setup_finished
  async def read_luminescence(self, focal_height: float, **backend_kwargs) -> List[List[float]]:
    """Read the luminescence from the plate.

    Args:
      focal_height: The focal height to read the luminescence at, in micrometers.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """
    plate = self.get_plate()
    extra_args = self._check_args(self.backend.read_luminescence, backend_kwargs, {"focal_height", "plate"})
    for arg in extra_args:
      backend_kwargs.pop(arg)
    return await self.backend.read_luminescence(focal_height=focal_height, plate=plate, **backend_kwargs)

  @need_setup_finished
  async def read_absorbance(self, wavelength: int, **backend_kwargs) -> List[List[float]]:
    """Read the absorbance from the plate in OD, unless otherwise specified by the backend.

    Args:
      wavelength: The wavelength to read the absorbance at, in nanometers.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """
    plate = self.get_plate()
    extra_args = self._check_args(self.backend.read_absorbance, backend_kwargs, {"wavelength", "plate"})
    for arg in extra_args:
      backend_kwargs.pop(arg)
    return await self.backend.read_absorbance(wavelength=wavelength, plate=plate, **backend_kwargs)

  @need_setup_finished
  async def read_fluorescence(
    self,
    excitation_wavelength: int,
    emission_wavelength: int,
    focal_height: float,
    **backend_kwargs,
  ) -> List[List[float]]:
    """Read the fluorescence from the plate.

    Args:
      excitation_wavelength: The excitation wavelength to read the fluorescence at, in nanometers.
      emission_wavelength: The emission wavelength to read the fluorescence at, in nanometers.
      focal_height: The focal height to read the fluorescence at, in micrometers.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """
    plate = self.get_plate()
    extra_args = self._check_args(self.backend.read_fluorescence, backend_kwargs, {"excitation_wavelength", "emission_wavelength", "focal_height", "plate"})
    for arg in extra_args:
      backend_kwargs.pop(arg)
    return await self.backend.read_fluorescence(
      excitation_wavelength=excitation_wavelength,
      emission_wavelength=emission_wavelength,
      focal_height=focal_height,
      plate=plate,
      **backend_kwargs,
    )
