"""Schema conformance tests — verify PCBData schema structure and unit normalization."""

from __future__ import annotations

from shared.schemas import (
    BoardThickness,
    CopperWeights,
    DrillRow,
    ImpedanceControl,
    ImpedanceRange,
    IPCClass,
    LayerSpec,
    PCBData,
    Silkscreen,
    SolderMask,
    normalize_units,
)


class TestPCBDataSchema:
    """Tests for PCBData schema structure."""

    def test_minimal_instance(self):
        """PCBData can be created with all defaults."""
        data = PCBData()
        assert data.ipc_specs == []
        assert data.layer_stackup == []
        assert data.drill_table == []

    def test_full_instance(self):
        """PCBData can be created with all fields populated."""
        data = PCBData(
            part_number="12345",
            manufacturer="Acme Corp",
            drawing_title="Test PCB",
            is_itar=False,
            ipc_class=IPCClass.CLASS_3,
            ipc_specs=["IPC-6012", "IPC-A-600"],
            layer_count=6,
            layer_stackup=[
                LayerSpec(number=1, function="signal"),
                LayerSpec(number=2, function="plane"),
            ],
            material="FR4",
            board_thickness=BoardThickness(
                nominal=0.062,
                plus_tol=0.005,
                minus_tol=0.005,
                unit="in",
            ),
            copper_weights=CopperWeights(
                signal_layers_oz=1.0,
                plane_layers_oz=1.0,
                external_finished_oz=2.0,
            ),
            surface_finish="ENIG",
            solder_mask=SolderMask(
                present=True,
                color="Green",
                type="LPI",
            ),
            silkscreen=Silkscreen(
                present=True,
                color="White",
            ),
            impedance_control=ImpedanceControl(
                controlled=True,
                single_ended=ImpedanceRange(min=50.0, max=50.0, unit="ohm"),
                trace_width_mils=5.0,
                layers=[1, 6],
            ),
            drill_table=[
                DrillRow(size_mils=10.0, qty=100, symbol="+", plated=True),
            ],
            fabrication_notes="Test notes",
        )

        assert data.layer_count == 6
        assert data.ipc_class == IPCClass.CLASS_3
        assert data.board_thickness.unit == "in"
        assert data.drill_table[0].size_mils == 10.0

    def test_ipc_class_serialization(self):
        """IPCClass should serialize to string value."""
        data = PCBData(ipc_class=IPCClass.CLASS_3)
        d = data.model_dump()
        assert d["ipc_class"] == "Class 3"

    def test_optional_defaults(self):
        """All optional fields default to None."""
        data = PCBData()
        assert data.part_number is None
        assert data.manufacturer is None
        assert data.drawing_title is None
        assert data.is_itar is None
        assert data.ipc_class is None
        assert data.layer_count is None
        assert data.material is None
        assert data.board_thickness is None
        assert data.surface_finish is None


class TestNormalizeUnits:
    """Tests for unit normalization."""

    def test_inches_unchanged(self):
        """Inches values should pass through unchanged."""
        data = PCBData(
            board_thickness=BoardThickness(nominal=0.062, plus_tol=0.005, minus_tol=0.005),
            drill_table=[DrillRow(size_mils=10.0)],
        )
        result = normalize_units(data)
        assert result.board_thickness.nominal == 0.062
        assert result.board_thickness.unit == "in"
        assert result.drill_table[0].size_mils == 10.0

    def test_mm_to_inches(self):
        """Board thickness > 1.0 is treated as mm and converted."""
        data = PCBData(
            board_thickness=BoardThickness(nominal=1.57, plus_tol=0.127, minus_tol=0.127),
        )
        result = normalize_units(data)
        assert abs(result.board_thickness.nominal - 1.57 / 25.4) < 0.001
        assert result.board_thickness.unit == "in"

    def test_drill_mils_in_normal_range_unchanged(self):
        """Drill sizes up to 1000 (e.g. mounting/tooling holes) pass through unchanged."""
        data = PCBData(
            drill_table=[DrillRow(size_mils=124.0)],  # a real mounting-hole size, not mm
        )
        result = normalize_units(data)
        assert result.drill_table[0].size_mils == 124.0

    def test_drill_microns_to_mils(self):
        """Drill sizes > 1000 are treated as microns and converted to mils."""
        data = PCBData(
            drill_table=[DrillRow(size_mils=2540.0)],  # 2540 microns = 100 mils
        )
        result = normalize_units(data)
        assert result.drill_table[0].size_mils == 2540.0 / 25.4

    def test_drill_raw_inches_to_mils(self):
        """Drill sizes < 1 are raw decimal inches never converted to mils."""
        data = PCBData(
            drill_table=[DrillRow(size_mils=0.024)],  # .024" -> 24 mils
        )
        result = normalize_units(data)
        assert result.drill_table[0].size_mils == 24.0

    def test_drill_all_identical_sizes_discarded(self):
        """A 3+ row table where every size is identical is discarded as degenerate."""
        data = PCBData(
            drill_table=[
                DrillRow(size_mils=39.3701, qty=1),
                DrillRow(size_mils=39.3701, qty=2),
                DrillRow(size_mils=39.3701, qty=3),
            ],
        )
        result = normalize_units(data)
        assert result.drill_table == []

    def test_drill_two_identical_sizes_kept(self):
        """Below the 3-row threshold, identical sizes are plausible and kept."""
        data = PCBData(
            drill_table=[
                DrillRow(size_mils=20.0, qty=1),
                DrillRow(size_mils=20.0, qty=2),
            ],
        )
        result = normalize_units(data)
        assert len(result.drill_table) == 2

    def test_trace_width_mm_to_mils(self):
        """Trace widths > 50 are treated as mm."""
        data = PCBData(
            impedance_control=ImpedanceControl(
                trace_width_mils=127.0,
            ),
        )
        result = normalize_units(data)
        # 127 is between 50-1000 so treated as mm
        assert result.impedance_control.trace_width_mils == 127.0 / 0.0254


class TestNestedModels:
    """Tests for nested model structures."""

    def test_board_thickness(self):
        """BoardThickness has correct defaults."""
        bt = BoardThickness()
        assert bt.unit == "in"
        assert bt.nominal is None

    def test_solder_mask(self):
        """SolderMask has correct defaults."""
        sm = SolderMask()
        assert sm.present is False

    def test_silkscreen(self):
        """Silkscreen has correct defaults."""
        sk = Silkscreen()
        assert sk.present is False

    def test_impedance_control(self):
        """ImpedanceControl has correct defaults."""
        ic = ImpedanceControl()
        assert ic.controlled is False
        assert ic.layers == []

    def test_drill_row(self):
        """DrillRow has all optional fields."""
        dr = DrillRow(size_mils=10.0)
        assert dr.size_mils == 10.0
        assert dr.qty is None
