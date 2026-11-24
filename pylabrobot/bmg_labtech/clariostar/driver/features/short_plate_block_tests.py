"""The short plate block of firmware 1.20 (an original CLARIOstar), against the RUNs MARS sent it.

On 1.20 the plate block has no `00` between the plate geometry and the well mask, and the
pre-separator is 8 bytes. These tests pin everything from the start of the payload through the
separator, the well-scan field and the settling byte to MARS's frames, with the plate geometry
replaced by the PLR plate's (as the other modes' tests do). The fields after the settling byte
are the modes' own and are not compared here: MARS's settings there were not recorded.
"""

import unittest
from typing import Any, Callable, TypeVar

from pylabrobot.bmg_labtech.clariostar.driver import master as master_module
from pylabrobot.bmg_labtech.clariostar.driver.configuration import DeviceConfiguration
from pylabrobot.bmg_labtech.clariostar.driver.configuration_tests import (
  CLARIOSTAR_1_20_INIT_PAYLOAD,
  CLARIOSTAR_1_20_PARAMETERS_PAYLOAD,
  decoded,
)
from pylabrobot.bmg_labtech.clariostar.driver.features import measurement as measurement_module
from pylabrobot.bmg_labtech.clariostar.driver.features.absorbance import Absorbance
from pylabrobot.bmg_labtech.clariostar.driver.features.absorbance_tests import (
  RUN_REPLY,
  STATUS_MEASURED,
  STATUS_MEASURING,
)
from pylabrobot.bmg_labtech.clariostar.driver.features.drawer_tests import make_plate_holder
from pylabrobot.bmg_labtech.clariostar.driver.features.fluorescence import Fluorescence
from pylabrobot.bmg_labtech.clariostar.driver.features.luminescence import Luminescence
from pylabrobot.bmg_labtech.clariostar.driver.features.luminescence_tests import (
  LUMINESCENCE_DATA_REPLY,
  plate,
)
from pylabrobot.bmg_labtech.clariostar.driver.features.measurement import (
  Measurement,
  MeasurementSettings,
  ShakeSettings,
)
from pylabrobot.bmg_labtech.clariostar.driver.master_tests import STATUS_IDLE, FakeFTDI, make_driver
from pylabrobot.bmg_labtech.protocol.framing import ftdi_extract_payload

# MARS's RUN payloads to an original CLARIOstar on firmware 1.20, from its communication log: full
# 96-well plate reads with MARS's settings, luminescence (top optic), fluorescence (bottom optic)
# and absorbance at 600 nm. Payload 63 is the scan byte, 64-71 the pre-separator, 72 the separator.
MARS_LUMINESCENCE_RUN = bytes.fromhex(
  "0431ec2166059604602c561d060c08ffffffffffffffffffffffff000000000000000000000000000000000000"
  "00000000000000000000000000000000000002010000002004001e270f270f01044c00000100000dac000100010001"
  "00010001000600000000000000000002000000000100000001006400010000"
)
MARS_FLUORESCENCE_RUN = bytes.fromhex(
  "0431ec2166059604602c561d060c08ffffffffffffffffffffffff000000000000000000000000000000000000"
  "00000000000000000000000000000000000002400000002004001e270f270f0a033e"
)
MARS_ABSORBANCE_RUN = bytes.fromhex(
  "0431ec2166059604602c561d060c08ffffffffffffffffffffffff000000000000000000000000000000000000"
  "00000000000000000000000000000000000002020000002004001e270f270f0501177000000064"
)

_M = TypeVar("_M", bound=Measurement)

# Through the separator and the settling byte (no well-scan field on a point read).
RUN_START_LENGTH = 77


def decoded_1_20() -> DeviceConfiguration:
  """The configuration an original CLARIOstar on firmware 1.20 decodes to."""
  configuration = DeviceConfiguration()
  configuration.update_from_init_reply(CLARIOSTAR_1_20_INIT_PAYLOAD)
  configuration.update_from_parameters(
    master_module.CLARIOstarDriver._decode_parameters(CLARIOSTAR_1_20_PARAMETERS_PAYLOAD)
  )
  return configuration


def make_feature(cls: Callable[[Any], _M], io: FakeFTDI, configuration: DeviceConfiguration) -> _M:
  """A `cls` measurement on a driver with `configuration`, the drawer holding `plate()`."""
  driver = make_driver(io)
  driver.configuration = configuration
  driver.drawer.plate_holder = make_plate_holder()
  driver.drawer.plate_holder.assign_child_resource(plate())
  return cls(driver)


def on_plate(mars_payload: bytes) -> bytes:
  """`mars_payload` from the RUN family byte through the settling byte, with `plate()`'s geometry."""
  payload = bytearray(mars_payload[:RUN_START_LENGTH])
  payload[1:13] = Measurement._encode_plate_geometry(plate())[:12]
  return bytes(payload)


# MARS's scan byte, `02`, under upstream's bit meanings: row by row, bidirectional, top left. This
# reader's meanings of the scan bits are not verified; `02` is the byte MARS sent.
MARS_SCAN = MeasurementSettings(reading_direction="row_major")


class ShortPlateBlockTests(unittest.IsolatedAsyncioTestCase):
  def run_start(
    self, cls: Callable[[Any], Measurement], settings: MeasurementSettings, **kwargs: Any
  ) -> bytes:
    feature = make_feature(cls, FakeFTDI(), decoded_1_20())
    p = plate()
    return b"\x04" + feature._compute_run_start(p, p.get_all_items(), settings, None, **kwargs)

  def test_luminescence_run_start_is_mars(self):
    settings = MeasurementSettings(reading_direction="row_major", settling_time=0.0)
    ours = self.run_start(Luminescence, settings, enhanced_dynamic_range=True)
    self.assertEqual(ours, on_plate(MARS_LUMINESCENCE_RUN))

  def test_fluorescence_bottom_optic_run_start_is_mars(self):
    settings = MeasurementSettings(reading_direction="row_major", settling_time=0.2)
    ours = self.run_start(Fluorescence, settings, optics=Fluorescence._OPTIC_POSITIONS["bottom"])
    self.assertEqual(ours, on_plate(MARS_FLUORESCENCE_RUN))

  def test_absorbance_run_start_is_mars(self):
    ours = self.run_start(Absorbance, MARS_SCAN)
    self.assertEqual(ours, on_plate(MARS_ABSORBANCE_RUN))

  async def test_a_luminescence_read_sends_the_short_block(self):
    io = FakeFTDI()
    io.replies = [
      STATUS_IDLE,
      RUN_REPLY,
      STATUS_MEASURING,
      STATUS_MEASURED,
      LUMINESCENCE_DATA_REPLY,
    ]
    feature = make_feature(Luminescence, io, decoded_1_20())
    settings = MeasurementSettings(reading_direction="row_major", settling_time=0.0)
    await feature.measure(focal_height=11.0, measurement_settings=settings)
    payload = ftdi_extract_payload(io.written[1])
    self.assertEqual(payload[:RUN_START_LENGTH], on_plate(MARS_LUMINESCENCE_RUN))
    self.assertEqual(payload[77:79], round(11.0 * 100).to_bytes(2, "big"))

  def test_the_plus_keeps_the_long_block(self):
    feature = make_feature(Luminescence, FakeFTDI(), decoded())
    p = plate()
    start = feature._compute_run_start(p, p.get_all_items(), MARS_SCAN, None)
    self.assertEqual(start[14], 0x00)  # the pad
    self.assertEqual(start[95:99], Measurement._SEPARATOR)

  def test_an_orbital_shake(self):
    # MARS's frames on this reader: 500 rpm for 300 s.
    feature = make_feature(Luminescence, FakeFTDI(), decoded_1_20())
    shake = ShakeSettings(pattern="orbital", speed=500, duration=300)
    self.assertEqual(
      feature._compute_short_pre_separator(MARS_SCAN, shake), bytes.fromhex("010000002104012c")
    )

  def test_other_shake_patterns_are_refused(self):
    feature = make_feature(Luminescence, FakeFTDI(), decoded_1_20())
    for pattern in ("linear", "double_orbital"):
      with self.assertRaisesRegex(ValueError, "no known encoding on firmware 1.20"):
        feature._compute_short_pre_separator(
          MARS_SCAN, ShakeSettings(pattern=pattern, speed=500, duration=30)
        )

  def test_the_scan_byte_is_the_plus_encoding(self):
    with self.assertNoLogs(measurement_module.logger, "WARNING"):
      start = self.run_start(Luminescence, MeasurementSettings())  # column-major
    self.assertEqual(start[63], 0x0A)

  def test_an_absorbance_spectrum_has_the_short_block(self):
    feature = make_feature(Absorbance, FakeFTDI(), decoded_1_20())
    p = plate()
    with self.assertNoLogs(measurement_module.logger, "WARNING"):
      params = feature._compute_spectrum_run_parameters(
        p, p.get_all_items(), 400, 700, 10, MARS_SCAN, None
      )
    self.assertEqual(params[71:75], Measurement._SEPARATOR)  # payload 72, without the family byte


if __name__ == "__main__":
  unittest.main()
