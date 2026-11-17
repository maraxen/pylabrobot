"""Injection in a luminescence read, against the RUNs MARS sent an original CLARIOstar (firmware 1.20).

MARS's frames below are luminescence endpoint reads it completed with one or two injections (orbital
well scan, a 300 s shake, NanoLuc's emission band). Upstream's luminescence reads only at each
well's centre, so these tests compare the RUN from the fields after the emission band to the end:
the normalization time, the trailer, the cycle time and the injection field. The fields before are
MARS's read settings, not injection's.
"""

import dataclasses
import unittest

from pylabrobot.bmg_labtech.clariostar.driver.features.absorbance_tests import (
  RUN_REPLY,
  STATUS_MEASURED,
  STATUS_MEASURING,
)
from pylabrobot.bmg_labtech.clariostar.driver.features.luminescence import Luminescence
from pylabrobot.bmg_labtech.clariostar.driver.features.luminescence_tests import (
  LUMINESCENCE_DATA_REPLY,
  plate,
)
from pylabrobot.bmg_labtech.clariostar.driver.features.measurement import MeasurementSettings
from pylabrobot.bmg_labtech.clariostar.driver.features.pumps import Injection, Pumps
from pylabrobot.bmg_labtech.clariostar.driver.features.pumps_tests import IDLE, IDLE_DOOR_OPEN
from pylabrobot.bmg_labtech.clariostar.driver.features.short_plate_block_tests import (
  MARS_LUMINESCENCE_RUN,
  decoded_1_20,
  make_feature,
)
from pylabrobot.bmg_labtech.clariostar.driver.master_tests import STATUS_IDLE, FakeFTDI
from pylabrobot.bmg_labtech.protocol.framing import ftdi_extract_payload

PUMPS_LOGGER = "pylabrobot.bmg_labtech.clariostar.driver.features.pumps"

_MARS_START = (
  "0431e82164059e04642c4a1d000c08ffffffffffffffffffffffff0000000000000000000000000000000000000000"
  "0000000000000000000000000000000002310000002104012c270f270f03010292000102ee00000100000dac0001"
  "0001000113e810d00005000000640000000000020000000001000000010008"
)
# MARS's RUN payloads (from the family byte), each with its injections and the cycle time MARS sent.
# Every one has an interval of 0.08 s and a normalization time of 1.00 s.
MARS_INJECTION_RUNS = [
  (  # pump 1, 5 uL
    bytes.fromhex(_MARS_START + "0152011104001e000000010000"),
    [Injection(pump=1, volume=5)],
    338,
  ),
  (  # pump 2, 1 uL
    bytes.fromhex(_MARS_START + "01510112040006000000010000"),
    [Injection(pump=2, volume=1)],
    337,
  ),
  (  # pump 1, 5 uL twice
    bytes.fromhex(_MARS_START + "0152021104001e0000000100001104001e000000010000"),
    [Injection(pump=1, volume=5), Injection(pump=1, volume=5)],
    338,
  ),
  (  # pump 1 then pump 2, 5 uL each
    bytes.fromhex(_MARS_START + "0159021104001e0000000100001204001e000000010000"),
    [Injection(pump=1, volume=5), Injection(pump=2, volume=5)],
    345,
  ),
]
# Payload offset of the trailer's byte 1: `02` in every MARS read on this reader, with and without
# injection; upstream sends `00`, which the reader reads with.
_TRAILER_BYTE_1 = 112


def tail_length(injections: int) -> int:
  """Bytes from after the emission band to the end of a RUN with `injections` injections."""
  return 4 + 2 + 4 + 16 + 10 * injections


def as_upstream(mars_payload: bytes) -> bytes:
  """`mars_payload` with the trailer byte upstream sends instead of MARS's."""
  payload = bytearray(mars_payload)
  payload[_TRAILER_BYTE_1] = 0x00
  return bytes(payload)


def make_injecting_luminescence(io: FakeFTDI) -> Luminescence:
  """A luminescence feature on a firmware 1.20 reader with two pumps."""
  configuration = decoded_1_20()
  configuration.pumps = ("XCalibur", "XCalibur", None)
  feature = make_feature(Luminescence, io, configuration)
  feature._driver.pumps = Pumps(feature._driver)
  return feature


class InjectionEncodingTests(unittest.TestCase):
  def setUp(self):
    self.feature = make_injecting_luminescence(FakeFTDI())
    assert self.feature._driver.pumps is not None
    self.pumps: Pumps = self.feature._driver.pumps

  def run_parameters(self, injections, cycle_time: int) -> bytes:
    p = plate()
    return b"\x04" + self.feature._compute_run_parameters(
      p,
      p.get_all_items(),
      MeasurementSettings(settling_time=0.0),
      None,
      emission_wavelength=470,
      emission_bandwidth=80,
      gain=3500,
      focal_height=7.5,
      normalization_time=1.0,
      measurement_interval=0.08,
      cycle_time=cycle_time,
      injections=self.pumps.encode_injections(injections),
    )

  def test_mars_frames_from_the_emission_band_on(self):
    for mars, injections, cycle_time in MARS_INJECTION_RUNS:
      with self.subTest(injections=injections):
        n = tail_length(len(injections))
        ours = self.run_parameters(injections, cycle_time)
        self.assertEqual(ours[-n:], as_upstream(mars)[-n:])

  def test_the_trailer_byte_mars_sends_is_not_injections(self):
    # MARS sends `02` in the trailer's byte 1 with injection and without (visit 2's read).
    for mars, _, _ in MARS_INJECTION_RUNS:
      self.assertEqual(mars[_TRAILER_BYTE_1], 0x02)
    # Visit 2's read ends `... 00 01 00 00`: the 1 s cycle time, then two `00` where upstream sends
    # one; its trailer's byte 1 is 16 bytes from the end.
    self.assertEqual(MARS_LUMINESCENCE_RUN[-16], 0x02)
    self.assertEqual(MARS_LUMINESCENCE_RUN[-4:], bytes.fromhex("00010000"))

  def test_each_field_changes_the_block(self):
    mars, injections, cycle_time = MARS_INJECTION_RUNS[0]
    n = tail_length(1)
    reference = as_upstream(mars)[-n:]
    (injection,) = injections
    for change in (
      {"pump": 2},
      {"volume": 3},
      {"flow_rate": 430},
      {"smart_dispensing": False},
      {"shake_duration": 300},
    ):
      with self.subTest(change=change):
        ours = self.run_parameters([dataclasses.replace(injection, **change)], cycle_time)
        self.assertNotEqual(ours[-n:], reference)
    self.assertNotEqual(self.run_parameters(injections, 1)[-n:], reference)

  def test_no_injection_is_upstreams_end(self):
    self.assertEqual(self.pumps.encode_injections([]), b"\x00")
    ours = self.run_parameters([], 1)
    self.assertEqual(ours[-3:], bytes.fromhex("000100"))

  def test_volume_is_six_times_ul(self):
    block = self.pumps.encode_injections([Injection(pump=1, volume=10)])
    self.assertEqual(block, bytes.fromhex("01" + "1104003c000000010000"))

  def test_settings_out_of_range_are_refused(self):
    for injections, message in (
      ([Injection(pump=1, volume=5)] * 3, "at most 2 injections"),
      ([Injection(pump=3, volume=5)], "pump must be one of"),
      ([Injection(pump=1, volume=0)], "volume must be"),
      ([Injection(pump=1, volume=0.1)], "volume must be"),
      ([Injection(pump=1, volume=5, flow_rate=200)], "flow_rate must be"),
      ([Injection(pump=1, volume=5, shake_duration=-1)], "shake_duration"),
    ):
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        self.pumps.encode_injections(injections)

  def test_an_unfitted_pump_is_refused(self):
    self.feature._driver.configuration.pumps = ("XCalibur", None, None)
    with self.assertRaisesRegex(ValueError, "pump 2 is not fitted"):
      self.pumps.encode_injections([Injection(pump=2, volume=5)])

  def test_settings_not_seen_in_a_completed_read_warn(self):
    with self.assertLogs(PUMPS_LOGGER, "WARNING") as logs:
      self.pumps.encode_injections([Injection(pump=1, volume=10)])
    self.assertIn("inferred", logs.output[0])
    with self.assertNoLogs(PUMPS_LOGGER, "WARNING"):
      self.pumps.encode_injections([Injection(pump=2, volume=3)])


class InjectingReadTests(unittest.IsolatedAsyncioTestCase):
  async def test_a_read_that_injects_sends_the_injection_field(self):
    io = FakeFTDI()
    io.replies = [
      IDLE_DOOR_OPEN,
      STATUS_IDLE,
      RUN_REPLY,
      STATUS_MEASURING,
      STATUS_MEASURED,
      LUMINESCENCE_DATA_REPLY,
    ]
    feature = make_injecting_luminescence(io)
    settings = MeasurementSettings(reading_direction="row_major", settling_time=0.0)
    await feature.measure(
      focal_height=11.0,
      measurement_settings=settings,
      injections=[Injection(pump=1, volume=10)],
      cycle_time=400,
    )
    payload = ftdi_extract_payload(io.written[2])
    self.assertEqual(payload[0], 0x04)
    self.assertEqual(payload[-13:], bytes.fromhex("0190" + "01" + "1104003c000000010000"))

  async def test_a_shut_reagent_door_sends_no_run(self):
    io = FakeFTDI()
    io.replies = [IDLE]
    feature = make_injecting_luminescence(io)
    with self.assertRaisesRegex(RuntimeError, "open the reagent door"):
      await feature.measure(
        measurement_settings=MeasurementSettings(settling_time=0.0),
        injections=[Injection(pump=1, volume=5)],
      )
    self.assertEqual(len(io.written), 1)  # the status query only
    self.assertNotEqual(ftdi_extract_payload(io.written[0])[0], 0x04)

  async def test_a_reader_without_pumps_cannot_inject(self):
    io = FakeFTDI()
    feature = make_feature(Luminescence, io, decoded_1_20())
    feature._driver.pumps = None
    with self.assertRaisesRegex(ValueError, "no injector pump"):
      await feature.measure(
        measurement_settings=MeasurementSettings(settling_time=0.0),
        injections=[Injection(pump=1, volume=5)],
      )
    self.assertEqual(io.written, [])

  async def test_a_cycle_time_out_of_range_sends_nothing(self):
    io = FakeFTDI()
    feature = make_injecting_luminescence(io)
    with self.assertRaisesRegex(ValueError, "cycle_time"):
      await feature.measure(
        measurement_settings=MeasurementSettings(settling_time=0.0), cycle_time=0
      )
    self.assertEqual(io.written, [])


if __name__ == "__main__":
  unittest.main()
