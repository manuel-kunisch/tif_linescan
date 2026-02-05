import numpy as np
from typing import List, Tuple
import pyqtgraph as pg
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from functools import partial

COLOR_TO_CMAP = {
    "tab:red": "Reds_r",
    "tab:green": "Greens_r",
    "tab:blue": "Blues_r",
    "tab:orange": "Oranges_r",
    "tab:purple": "Purples_r",
    "tab:brown": "copper_r",
    "tab:pink": "pink_r",
    "tab:gray": "Greys_r",
    "tab:cyan": "winter_r",
    "tab:olive": "YlGn_r",
    # fallback
    "red": "Reds_r",
    "green": "Greens_r",
    "blue": "Blues_r",
    "orange": "Oranges_r",
    "purple": "Purples_r",
    "brown": "copper_r",
}

class LineScanPlotSaver(QtWidgets.QDialog):
    """
    A dialog for saving line scan plots with matplotlib with various customization options.
    """
    def __init__(self, image_stack: np.ndarray, roi: pg.LineSegmentROI,
                 profile_coords: List[np.ndarray], pixel_size_um: float = 1.0,
                 line_colors: List[str] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save LineScan Plot")
        self.setMinimumWidth(600)

        self.stack = image_stack  # shape: (C, Y, X)
        self.roi = roi
        self.coords = profile_coords
        self.pixel_size_um = pixel_size_um
        self.channel = 0
        # --- z‑stack support ------------------------------------------------
        self.z_size = self.stack.shape[0]
        self.z_index = 0
        self.scalebar_um = 50.0
        self.cmap = 'gnuplot'
        self.line_colors = line_colors if line_colors else ["white"] * self.stack.shape[0]

        # contrast limits
        self.vmin = None
        self.vmax = None

        self.line_color = QtGui.QColor("white")
        self.line_width = 1.5
        self.line_style = 'solid'
        self.font_size = 10
        self.show_scalebar = True
        self.scalebar_position = "bottom-left"

        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Matplotlib canvas
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        # Controls layout
        controls = QtWidgets.QGridLayout()

        # Channel selector
        self.channel_selector = QtWidgets.QSpinBox()
        self.channel_selector.setMaximum(self.stack.shape[1] - 1)
        self.channel_selector.setPrefix("Channel ")
        self.channel_selector.valueChanged.connect(self._on_channel_changed)
        controls.addWidget(self.channel_selector, 0, 0, 1, 2)

        # Z‑slice selector (only visible for 4‑D stacks)
        if self.z_size > 1:
            self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(self.z_size - 1)
            self.z_slider.valueChanged.connect(self._on_z_slider)
            controls.addWidget(QtWidgets.QLabel("Z:"), 0, 2)
            controls.addWidget(self.z_slider, 0, 3)

            self.z_spin = QtWidgets.QSpinBox()
            self.z_spin.setRange(0, self.z_size - 1)
            self.z_spin.valueChanged.connect(self._on_z_spin)
            controls.addWidget(self.z_spin, 0, 4)

        # Scalebar size
        self.scalebar_input = QtWidgets.QDoubleSpinBox()
        self.scalebar_input.setDecimals(1)
        self.scalebar_input.setSuffix(" µm")
        self.scalebar_input.setValue(self.scalebar_um)
        self.scalebar_input.valueChanged.connect(self._on_scalebar_changed)
        controls.addWidget(QtWidgets.QLabel("Scalebar:"), 1, 0)
        controls.addWidget(self.scalebar_input, 1, 1)

        # Scalebar font size
        self.font_input = QtWidgets.QSpinBox()
        self.font_input.setValue(self.font_size)
        self.font_input.setMinimum(1)
        self.font_input.setMaximum(30)
        self.font_input.valueChanged.connect(self._on_font_size_changed)
        controls.addWidget(QtWidgets.QLabel("Font size:"), 2, 0)
        controls.addWidget(self.font_input, 2, 1)

        # Toggle scalebar
        self.scalebar_toggle = QtWidgets.QCheckBox("Show Scalebar")
        self.scalebar_toggle.setChecked(True)
        self.scalebar_toggle.stateChanged.connect(self._on_scalebar_toggle)
        controls.addWidget(self.scalebar_toggle, 3, 0, 1, 2)

        # Scalebar position
        self.scalebar_pos_dropdown = QtWidgets.QComboBox()
        self.scalebar_pos_dropdown.addItems(["bottom-left", "bottom-right", "top-left", "top-right"])
        self.scalebar_pos_dropdown.currentTextChanged.connect(self._on_scalebar_pos_changed)
        controls.addWidget(QtWidgets.QLabel("Scalebar position:"), 4, 0)
        controls.addWidget(self.scalebar_pos_dropdown, 4, 1)

        # Line color and thickness
        self.line_thickness_input = QtWidgets.QDoubleSpinBox()
        self.line_thickness_input.setDecimals(1)
        self.line_thickness_input.setMinimum(0.1)
        self.line_thickness_input.setValue(self.line_width)
        self.line_thickness_input.valueChanged.connect(self._on_line_thickness_changed)
        controls.addWidget(QtWidgets.QLabel("Line Width:"), 5, 0)
        controls.addWidget(self.line_thickness_input, 5, 1)

        self.line_color_btn = QtWidgets.QPushButton("Pick Line Color")
        self.line_color_btn.clicked.connect(self._on_pick_line_color)
        controls.addWidget(self.line_color_btn, 6, 0, 1, 2)

        # ---- vmin / vmax sliders ---------------------------------------
        row = 7
        self.vmin_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vmin_slider.setMinimum(0); self.vmin_slider.setMaximum(1000)
        self.vmin_slider.setValue(0)
        self.vmin_slider.valueChanged.connect(self._on_vmin_slider)
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setDecimals(0)
        self.vmin_spin.valueChanged.connect(self._on_vmin_spin)
        controls.addWidget(QtWidgets.QLabel("vmin:"), row, 0)
        controls.addWidget(self.vmin_slider, row, 1)
        controls.addWidget(self.vmin_spin, row, 2)

        row += 1
        self.vmax_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vmax_slider.setMinimum(0); self.vmax_slider.setMaximum(1000)
        self.vmax_slider.setValue(1000)
        self.vmax_slider.valueChanged.connect(self._on_vmax_slider)
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setDecimals(0)
        self.vmax_spin.valueChanged.connect(self._on_vmax_spin)
        controls.addWidget(QtWidgets.QLabel("vmax:"), row, 0)
        controls.addWidget(self.vmax_slider, row, 1)
        controls.addWidget(self.vmax_spin, row, 2)


        # Colormap dropdown
        row += 1
        self.cmap_dropdown = QtWidgets.QComboBox()
        self.cmap_dropdown.addItems(sorted(plt.colormaps()))
        self.cmap_dropdown.setCurrentText(self.cmap)
        self.cmap_dropdown.currentTextChanged.connect(self._on_cmap_changed)
        controls.addWidget(QtWidgets.QLabel("Colormap:"), row, 0)
        controls.addWidget(self.cmap_dropdown, row, 1)

        # Save buttons
        row += 1
        self.save_btn = QtWidgets.QPushButton("Save Current")
        self.save_btn.clicked.connect(self._save_figure)
        controls.addWidget(self.save_btn, row, 0)

        self.save_all_btn = QtWidgets.QPushButton("Save All Channels")
        self.save_all_btn.clicked.connect(self._save_all_channels)
        controls.addWidget(self.save_all_btn, row, 1)

        layout.addLayout(controls)

    def _on_channel_changed(self, val):
        self.channel = val
        self._update_plot()

    def _on_z_slider(self, val):
        self.z_index = val
        self.z_spin.blockSignals(True)
        self.z_spin.setValue(val)
        self.z_spin.blockSignals(False)
        self._update_plot()

    def _on_z_spin(self, val):
        self.z_index = val
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(val)
        self.z_slider.blockSignals(False)
        self._update_plot()

    def _on_scalebar_changed(self, val):
        self.scalebar_um = val
        self._update_plot()

    def _on_font_size_changed(self, val):
        self.font_size = val
        self._update_plot()

    def _on_scalebar_toggle(self, state):
        self.show_scalebar = bool(state)
        self._update_plot()

    def _on_scalebar_pos_changed(self, val):
        self.scalebar_position = val
        self._update_plot()

    def _on_line_thickness_changed(self, val):
        self.line_width = val
        self._update_plot()

    def _on_pick_line_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.line_color = color
            self._update_plot()

    def _on_cmap_changed(self, val):
        self.cmap = val
        self._update_plot()

    # --- vmin/vmax handlers ---------------------------------------------
    def _on_vmin_slider(self, val):
        self.vmin = val
        self.vmin_spin.blockSignals(True)
        self.vmin_spin.setValue(val)
        self.vmin_spin.blockSignals(False)
        self._update_plot()

    def _on_vmin_spin(self, val):
        self.vmin = val
        self.vmin_slider.blockSignals(True)
        self.vmin_slider.setValue(int(val))
        self.vmin_slider.blockSignals(False)
        self._update_plot()

    def _on_vmax_slider(self, val):
        self.vmax = val
        self.vmax_spin.blockSignals(True)
        self.vmax_spin.setValue(val)
        self.vmax_spin.blockSignals(False)
        self._update_plot()

    def _on_vmax_spin(self, val):
        self.vmax = val
        self.vmax_slider.blockSignals(True)
        self.vmax_slider.setValue(int(val))
        self.vmax_slider.blockSignals(False)
        self._update_plot()


    def _get_scalebar_position(self, bar_px, img_shape):
        margin = 5
        if self.scalebar_position == "bottom-left":
            return (margin, img_shape[0] - margin - 10)
        elif self.scalebar_position == "bottom-right":
            return (img_shape[1] - margin - bar_px, img_shape[0] - margin - 10)
        elif self.scalebar_position == "top-left":
            return (margin, margin)
        elif self.scalebar_position == "top-right":
            return (img_shape[1] - margin - bar_px, margin)
        return (margin, img_shape[0] - margin - 10)

    def _update_plot(self):
        self.ax.clear()
        if self.stack.ndim == 4:
            img = self.stack[self.z_index, self.channel]
        else:
            img = self.stack[self.channel]
        # update vmin/vmax sliders/spins if needed
        if self.vmin is None or self.vmax is None:
            self.vmin, self.vmax = float(img.min()), float(img.max())
            self.vmin_slider.setRange(int(self.vmin), int(self.vmax))
            self.vmax_slider.setRange(int(self.vmin), int(self.vmax))
            self.vmin_spin.setRange(self.vmin, self.vmax)
            self.vmax_spin.setRange(self.vmin, self.vmax)
            self.vmin_slider.setValue(int(self.vmin))
            self.vmax_slider.setValue(int(self.vmax))
            self.vmin_spin.setValue(self.vmin)
            self.vmax_spin.setValue(self.vmax)
        self.ax.imshow(
            img,
            cmap=self.cmap,
            origin='upper',
            vmin=self.vmin if self.vmin is not None else None,
            vmax=self.vmax if self.vmax is not None else None
        )

        # ROI line overlay
        self.ax.plot([p[0] for p in self.coords],
                     [p[1] for p in self.coords],
                     color=self.line_color.name(), linewidth=self.line_width, ls= self.line_style)

        if self.show_scalebar:
            # Scalebar overlay
            bar_px = self.scalebar_um / self.pixel_size_um
            x, y = self._get_scalebar_position(bar_px, img.shape)
            self.ax.add_patch(Rectangle((x, y), bar_px, 2, color='white'))
            self.ax.text(x + bar_px / 2, y - 5,
                         f"{self.scalebar_um:.0f} µm",
                         color='white', ha='center', va='bottom', fontsize=self.font_size)

        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0)
        self.canvas.draw()

    def _save_figure(self):
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "linescan.png",
                                                            "PNG Files (*.png)")
        if out_path:
            self.ax.set_position([0, 0, 1, 1])
            self.fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            print(f"Saved to: {out_path}")

    def _save_all_channels(self):
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Save All Channels")
        if not out_dir:
            return

        original_channel = self.channel
        original_cmap = self.cmap

        for ch in range(self.stack.shape[1]):
            self.channel = ch
            line_color = self.line_colors[ch]
            mapped_cmap = COLOR_TO_CMAP.get(line_color, 'gray')
            self.cmap = mapped_cmap  # override colormap

            self._update_plot()
            out_path = f"{out_dir}/linescan_ch{ch}.png"
            self.ax.set_position([0, 0, 1, 1])
            self.fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            print(f"Saved to: {out_path}")

        # Restore original state
        self.channel = original_channel
        self.cmap = original_cmap
        self._update_plot()
