"""
Interactive multi‑channel line‑scan viewer
=========================================

*   **Display** a (C, Y, X) TIFF stack with physical‑unit axes using the
    *PhysicalUnitsImageView* class (PyQtGraph).
*   **Draw** a line (pg.LineSegmentROI) and get a live intensity profile for
    **all channels** plotted with Matplotlib in their individual colours.
*   **Save** the current profile plot as PNG/PDF/SVG.
*   **Adjust** image field‑of‑view (FOV) or pixel size; scroll through channels.

Requirements
------------
```bash
pip install pyqt5 pyqtgraph matplotlib tifffile scikit-image numpy
```

Run it with `python linescan_app.py`.
"""

import json
import sys
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyqtgraph as pg
import tifffile as tiff
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyqtgraph.exporters import ImageExporter
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from linescan_plotter_mpl import LineScanPlotSaver
# --------------------------------------------------------------------------------------
#  PhysicalUnitsImageView  (the user already has this class; we re‑import if available)
# --------------------------------------------------------------------------------------
from physical_units_image_view import PhysicalUnitsImageView  # type: ignore
from scalebar import ScaleBar


tab_to_hex = {"tab:blue" : "#1f77b4",
                "tab:orange" : "#ff7f0e",
                "tab:green" : "#2ca02c",
                "tab:red" : "#d62728",
                "tab:purple" : "#9467bd",
                "tab:brown" : "#8c564b",
                "tab:pink" : "#e377c2",
                "tab:gray" : "#7f7f7f",
                "tab:olive" : "#bcbd22",
                "tab:cyan" : "#17becf"}
# --------------------------------------------------------------------------------------
#  ChannelColorDialog
# --------------------------------------------------------------------------------------
class ChannelColorDialog(QtWidgets.QDialog):
    """
    Small dialog that lets the user pick a Matplotlib / Qt colour
    for every channel and optionally save/load presets (JSON).
    """
    def __init__(self, colours: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel colours")
        self.resize(300, 200)
        self._colours = list(colours)       # copy *as a list*

        layout = QtWidgets.QGridLayout(self)

        # channel colour pickers
        for i, col in enumerate(self._colours):
            btn = QtWidgets.QPushButton()
            qcol = QtGui.QColor(col)
            if col.startswith("tab:"):
                # remove the 'tab:' prefix for PyQt6 compatibility
                qcol = QtGui.QColor(tab_to_hex.get(col, col[4:]))
            btn.setStyleSheet(f"background-color:{qcol.name()};")
            btn.clicked.connect(partial(self.pick_colour, i, btn))
            layout.addWidget(QtWidgets.QLabel(f"Ch {i}"), i, 0)
            layout.addWidget(btn, i, 1)

        # save / load preset buttons
        btn_save = QtWidgets.QPushButton("Save preset")
        btn_load = QtWidgets.QPushButton("Load preset")
        btn_save.clicked.connect(self.save_preset)
        btn_load.clicked.connect(self.load_preset)
        layout.addWidget(btn_save, len(self._colours), 0)
        layout.addWidget(btn_load, len(self._colours), 1)

        # OK / Cancel buttons (works for both PyQt6 & PyQt5)
        try:
            std = QtWidgets.QDialogButtonBox.StandardButton
            bb = QtWidgets.QDialogButtonBox(std.Ok | std.Cancel)
        except AttributeError:
            # older PyQt5 / PySide2
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                            QtWidgets.QDialogButtonBox.Cancel)

        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb, len(self._colours) + 1, 0, 1, 2)

    # ------------------------------------------------------------------
    def pick_colour(self, idx: int, btn: QtWidgets.QPushButton):
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._colours[idx]), self)
        if colour.isValid():
            self._colours[idx] = colour.name()
            btn.setStyleSheet(f"background-color:{colour.name()};")

    def save_preset(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save preset", "colours.json", "JSON (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump(self._colours, fh)
            QtWidgets.QMessageBox.information(self, "Saved", f"Preset saved to {path}")

    def load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) >= len(self._colours):
                self._colours = data[:len(self._colours)]
                # update button styles
                for row in range(len(self._colours)):
                    btn: QtWidgets.QPushButton = self.layout().itemAtPosition(row, 1).widget()
                    btn.setStyleSheet(f"background:{self._colours[row]}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))

    # ------------------------------------------------------------------
    def colours(self) -> list[str]:
        return self._colours


#  Matplotlib canvas
# --------------------------------------------------------------------------------------
class ProfileCanvas(FigureCanvas):
    """Matplotlib canvas that shows the multi‑channel line profile."""

    colours = (
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown",
    )

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Distance (µm)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.figure.tight_layout()
        self.lines: List[Tuple] = []
        self.labels : List[str] = []  # labels for each channel

        # set properties for the canvas
        self.fig.tight_layout()
        # font
        self.ax.tick_params(labelsize=11)
        self.ax.xaxis.label.set_size(11)
        self.ax.yaxis.label.set_size(11)
        self.ax.set_title("Intensity Profile", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend(fontsize=11)
        self.figure.canvas.mpl_connect('resize_event', self.on_resize)
        self.figure.set_dpi(100)  # set DPI for better resolution

    def on_resize(self, event):
        """Handle canvas resize events."""
        # Re-draw the canvas to ensure it fits the new size
        self.figure.tight_layout()
        self.draw_idle()

    def plot_profiles(self, profiles: List[np.ndarray], coords = None):
        # get the previosu x and y labels
        xlabel = self.ax.get_xlabel()
        ylabel = self.ax.get_ylabel()
        # clear the previous lines
        self.ax.clear()
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        for c, prof in enumerate(profiles):
            color = self.colours[c % len(self.colours)]
            if coords is None:
                # create a default x-coordinates array if not provided
                coords = np.arange(len(prof))
            self.ax.plot(coords, prof, label=f"Ch {c}", color=color)
        self.ax.legend(self.labels if self.labels else [f"Channel{c}" for c in range(len(profiles))])
        self.draw_idle()

    def save_plot(self, path: Path):
        self.figure.savefig(path)

# --------------------------------------------------------------------------------------
#  Main application window
# --------------------------------------------------------------------------------------
class LineScanApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi‑channel Line‑Scan Viewer")
        self.resize(1200, 500)

        # central splitter --------------------------------------------------
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # image view --------------------------------------------------------
        self.image_view = PhysicalUnitsImageView(width=1, height=1, physical_unit="µm")
        splitter.addWidget(self.image_view)

        # matplotlib canvas -------------------------------------------------
        self.profile_canvas = ProfileCanvas()
        splitter.addWidget(self.profile_canvas)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ROI line ----------------------------------------------------------
        self.roi = pg.LineSegmentROI([[0, 0], [50, 0]], pen=pg.mkPen("yellow", width=2))
        self.image_view.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_profile)

        # data holders ------------------------------------------------------
        self.stack: np.ndarray | None = None        # (C, Y, X)
        self.stack_original: np.ndarray | None = None        # (C, Y, X)
        self.current_channel = 0

        self.saver: LineScanPlotSaver | None = None  # for saving line scan plots

        self.scalebar = None
        self.scalebar_visible = False
        self.scalebar_length_um = 50.0

        # toolbar -----------------------------------------------------------
        self._build_toolbar()
    # ------------------------------------------------------------------
    #  UI helpers
    # ------------------------------------------------------------------
    def _build_toolbar(self):
        tb = self.addToolBar("Main")

        # load image
        load_act = QtGui.QAction("Load Image", self)
        load_act.triggered.connect(self.load_image)
        tb.addAction(load_act)

        # save plot
        save_act = QtGui.QAction("Save Profile", self)
        save_act.triggered.connect(self.save_profile)
        tb.addAction(save_act)

        save_linescan_act = QtGui.QAction("Save Linescan", self)
        save_linescan_act.triggered.connect(self.save_plot)
        tb.addAction(save_linescan_act)

        tb.addSeparator()


        # channel selector
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setPrefix("Channel ")
        self.channel_spin.setMinimum(0)
        self.channel_spin.valueChanged.connect(self.change_channel)
        tb.addWidget(self.channel_spin)

        # after the channel spin-box
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setEnabled(False)
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setEnabled(False)

        self.z_slider.valueChanged.connect(self._on_z_slider)
        self.z_spin.valueChanged.connect(self._on_z_spin)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Z:"))
        tb.addWidget(self.z_slider)
        tb.addWidget(self.z_spin)

        tb.addSeparator()
        # field of view / pixel size inputs
        self.fov_w_edit = QtWidgets.QDoubleSpinBox()
        self.fov_w_edit.setDecimals(1)
        self.fov_w_edit.setSuffix(" µm  FOV‑W")
        self.fov_w_edit.setMaximum(np.inf)
        self.fov_w_edit.setValue(397)
        self.fov_h_edit = QtWidgets.QDoubleSpinBox()
        self.fov_h_edit.setDecimals(1)
        self.fov_h_edit.setSuffix(" µm  FOV‑H")
        self.fov_h_edit.setMaximum(np.inf)
        self.fov_h_edit.setValue(397)
        apply_fov = QtGui.QAction("Apply FOV", self)
        apply_fov.triggered.connect(self.apply_fov)
        self.apply_fov()  # initial FOV application


        tb.addWidget(self.fov_w_edit)
        tb.addWidget(self.fov_h_edit)
        tb.addAction(apply_fov)

        tb.addSeparator()
        # add checkbox to swap between physical units and pixels
        self.show_unit_cb = QtWidgets.QCheckBox("Show physical units")
        self.show_unit_cb.setChecked(True)
        self.show_unit_cb.stateChanged.connect(lambda x: self.update_profile())
        self.show_unit_cb.stateChanged.connect(lambda state: self.update_units(state))
        tb.addWidget(self.show_unit_cb)

        tb.addSeparator()
        # add checkbox to toggle normalization of the image
        self.normalize_cb = QtWidgets.QCheckBox("Normalize image")
        self.normalize_cb.setChecked(False)
        self.normalize_cb.stateChanged.connect(lambda x: self.normalize_channels(x))
        tb.addWidget(self.normalize_cb)


        tb.addSeparator()
        self.correlate_cb = QtWidgets.QCheckBox("Correlate channels")
        self.correlate_cb.setChecked(False)
        self.correlate_cb.stateChanged.connect(lambda x: self.correlate_channels(x))
        tb.addWidget(self.correlate_cb)

        tb.addSeparator()
        # add action to change channel names
        change_labels_act = QtGui.QAction("Change Channel Labels", self)
        change_labels_act.triggered.connect(self.change_channel_labels)
        tb.addAction(change_labels_act)

        colour_act = QtGui.QAction("Channel Colours", self)
        colour_act.triggered.connect(self.open_channel_colour_dialog)
        tb.addAction(colour_act)

        tb.addSeparator()
        # add check to either show default or channel specific colormaps
        self.channel_colormap_cb = QtWidgets.QCheckBox("Use channel colormaps")
        self.channel_colormap_cb.setChecked(False)
        self.channel_colormap_cb.stateChanged.connect(lambda check: self.change_channel(self.current_channel))
        tb.addWidget(self.channel_colormap_cb)


        tb.addSeparator()
        # add a show composite checkbox
        self.show_composite_cb = QtWidgets.QCheckBox("Show composite")
        self.show_composite_cb.setChecked(False)
        self.show_composite_cb.stateChanged.connect(lambda x: self.show_composite(x))
        tb.addWidget(self.show_composite_cb)

        # --- scalebar controls -------------------------------------------
        self.scalebar_cb = QtWidgets.QCheckBox("Show scalebar")
        self.scalebar_cb.setChecked(self.scalebar_visible)
        self.scalebar_cb.stateChanged.connect(lambda s: setattr(self, "scalebar_visible", bool(s)))
        self.scalebar_cb.stateChanged.connect(self.update_scalebar)
        tb.addWidget(self.scalebar_cb)
        self.update_scalebar()

        self.scalebar_spin = QtWidgets.QDoubleSpinBox()
        self.scalebar_spin.setSuffix(" µm")
        self.scalebar_spin.setDecimals(1)
        self.scalebar_spin.setRange(1, 1000)
        self.scalebar_spin.setValue(self.scalebar_length_um)
        self.scalebar_spin.valueChanged.connect(lambda v: setattr(self, "scalebar_length_um", v))
        self.scalebar_spin.valueChanged.connect(self.update_scalebar)
        tb.addWidget(self.scalebar_spin)

    def _on_z_slider(self, val):
        if val != self.z_index:
            self.z_index = val
            self.z_spin.blockSignals(True);
            self.z_spin.setValue(val);
            self.z_spin.blockSignals(False)
            self.update_display();
            self.update_profile()

    def _on_z_spin(self, val):
        if val != self.z_index:
            self.z_index = val
            self.z_slider.blockSignals(True);
            self.z_slider.setValue(val);
            self.z_slider.blockSignals(False)
            self.update_display();
            self.update_profile()

    # ------------------------------------------------------------------
    #  Slots
    # ------------------------------------------------------------------
    def load_image(self):
        """
        Returns the 4d image stack from a TIFF file.
        Order is (Z, C, Y, X) depending on the file.
        :return:
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF files (*.tif *.tiff)")
        if not path:
            return
        img = tiff.imread(path)
        if img.ndim == 2:  # → (1,1,Y,X)
            img = img[None, None, ...]
        elif img.ndim == 3:  # (C,Y,X) → (1,C,Y,X)
            img = img[None, ...]
        elif img.ndim != 4:  # anything else = error
            QtWidgets.QMessageBox.warning(self, "Error", "Unsupported dimensionality")
            return
        # get fiji metadata such as label, pixel size, etc.
        with tiff.TiffFile(path) as tif:
            if hasattr(tif, "pages") and tif.pages:
                page = tif.pages[0]
                if hasattr(page, "tags"):
                    tags = page.tags
                    if "ImageDescription" in tags:
                        description = tags["ImageDescription"].value
                        print(f"Image description: {description}")
                    if "PixelSize" in tags:
                        pixel_size = tags["PixelSize"].value
                        print(f"Pixel size: {pixel_size}")
                print(f'{tif.imagej_metadata.keys()=}')
                try:
                    label = tif.imagej_metadata['Labels']
                    print(len(label), "labels found in metadata")
                    # split each string after the \n separator and only keep the first part
                    self.profile_canvas.labels = [l.split('\n')[0] for l in label]
                except KeyError:
                    print("No labels found in metadata")

        # shape: (C, Y, X)
        self.stack_original = img.astype(np.uint16)
        self.z_size = img.shape[0]
        self.z_slider.setMaximum(self.z_size - 1)
        self.z_spin.setMaximum(self.z_size - 1)
        self.z_slider.setEnabled(self.z_size > 1)
        self.z_spin.setEnabled(self.z_size > 1)
        self.stack = self.stack_original.copy()  # keep original for later use
        self.channel_spin.setMaximum(self.stack.shape[0] - 1)

        self.current_channel = 0

        # --- z-stack support ---------------------------------------------
        self.z_size: int = 1  # number of z planes
        self.z_index: int = 0  # active z

        self.display_current_channel()
        self.image_view.autoRange()

        self.update_profile()
        self.profile_canvas.figure.tight_layout()
        self.profile_canvas.draw()

    def change_channel(self, idx: int):
        self.current_channel = idx
        self.display_current_channel()
        if self.channel_colormap_cb.isChecked():
            # set the colormap for the current channel to the same color as the profile line
            color = self.profile_canvas.colours[idx % len(self.profile_canvas.colours)]
            # remove the tab: prefix if it exists
            if color.startswith("tab:"):
                color = color[4:]
            # get the colormap from the color string
            color = pg.mkColor(color).getRgb()
            # set the upper handle of the histogram to the current channel color
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 2), color=np.array([[0, 0, 0, 255], color]))
            self.image_view.setColorMap(cmap)
        else:
            # get default fire colormap
            # (or any other default colormap you prefer)
            self.image_view.setColorMap(pg.colormap.getFromMatplotlib('viridis'))  # default fire colormap

    def show_composite(self, state=True):
        # show or hide the composite image
        if state:
            # lock the channel spinbox
            self.channel_spin.setEnabled(False)

            composite = np.zeros((self.stack.shape[1], self.stack.shape[2], 3), dtype=np.float32)
            for c in range(self.stack.shape[0]):
                ch_img = (self.stack[self.z_index, c] if self.stack.ndim == 4 else self.stack[c]).astype(np.float32)
                ch_img = (ch_img - ch_img.min()) / (np.ptp(ch_img) + np.finfo(float).eps)
                color = self.profile_canvas.colours[c % len(self.profile_canvas.colours)]
                if color.startswith("tab:"):
                    color = color[4:]
                color = pg.mkColor(color).getRgbF()[:3]
                for i in range(3):
                    composite[..., i] += ch_img * color[i]
            composite = np.clip(composite, 0, 1)

            # directly set image without using ImageView.setImage()
            self.image_view.imageItem.setImage(composite)
            self.image_view.setLevels(0, 1 )  # set levels for the composite image
        else:
            self.channel_spin.setEnabled(True)
            # display the current channel image
            self.display_current_channel()
            self.image_view.autoLevels()

    def apply_fov(self):
        w = self.fov_w_edit.value()
        h = self.fov_h_edit.value()
        self.image_view.adjust_dimensions(w, h)

    def save_profile(self):
        path, fmt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save profile", "profile.png", "Images (*.png *.pdf *.svg)")
        if path:
            self.profile_canvas.save_plot(Path(path))

    def save_image_view_with_scalebar(self):
        """
        Export the contents of the ImageView's ViewBox (no axes, no margins)
        as a PNG, including the ROI line and a scalebar if enabled.
        """
        if self.stack is None:
            return

        # Ask target file
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PNG", "image.png", "PNG Files (*.png)")
        if not fname:
            return

        vb: pg.ViewBox = self.image_view.view.getViewBox()

        # ---------- optional scalebar (using existing helper) ----------
        scalebar_item = None
        if self.scalebar_visible:
            px_size = self.fov_w_edit.value() / self.stack.shape[2]  # µm / px
            scalebar_item = ScaleBar(vb,
                                     px_size,
                                     scale_bar_size=self.scalebar_length_um,
                                     unit='µm',
                                     pen=pg.mkPen('w', width=2))

        # ---------- store current state --------------------------------
        # Fit ViewBox exactly to the image, no padding, correct Y‑flip
        h, w = self.stack.shape[1:]        # Y, X
        vb.setAspectLocked(True)
        vb.setRange(xRange=(0, w),
                    yRange=(h, 0), padding=0)

        QtCore.QCoreApplication.processEvents()  # ensure render

        # ---------- export only the ViewBox ----------------------------
        exporter = ImageExporter(vb)
        exporter.parameters()['width']  = w
        exporter.parameters()['height'] = h

        # hide the ticks
        exporter.parameters()['removeAxisItems'] = True
        exporter.export(fname)


        print(f"✅ Saved PNG to {fname}")

    def save_plot(self):
        """ save the plot with a linescan by opening a new interactive matplotlib window"""
        if self.stack is None:
            return

        # Get ROI pixel coordinates
        handles = self.roi.getSceneHandlePositions()
        view_obj = self.image_view.getView()
        view_box = view_obj.getViewBox() if hasattr(view_obj, "getViewBox") else view_obj
        p0 = view_box.mapSceneToView(handles[0][1])
        p1 = view_box.mapSceneToView(handles[1][1])

        x0 = self.image_view.x_mm_to_px_x(p0.x()) if hasattr(self.image_view, "x_mm_to_px_x") else p0.x()
        y0 = self.image_view.y_mm_to_px_y(p0.y()) if hasattr(self.image_view, "y_mm_to_px_y") else p0.y()
        x1 = self.image_view.x_mm_to_px_x(p1.x()) if hasattr(self.image_view, "x_mm_to_px_x") else p1.x()
        y1 = self.image_view.y_mm_to_px_y(p1.y()) if hasattr(self.image_view, "y_mm_to_px_y") else p1.y()

        # Sample evenly along the line
        N = 100
        x_coords = np.linspace(x0, x1, N)
        y_coords = np.linspace(y0, y1, N)
        coords = np.vstack([x_coords, y_coords]).T

        # Estimate pixel size (in µm) – fallback to 1.0 if unknown
        pixel_size = getattr(self.image_view, 'pixel_size_x', 1.0)
        if pixel_size is None:
            pixel_size = 1.0

        self.saver = LineScanPlotSaver(self.stack, self.roi, coords, pixel_size_um=pixel_size, line_colors=self.profile_canvas.colours,
                                       parent=self)
        self.saver.setWindowTitle("Save Linescan Plot")
        self.saver.setWindowIcon(QtGui.QIcon("icons/plot.png"))
        self.saver.show()

    # ------------------------------------------------------------------
    #  Core functionality
    # ------------------------------------------------------------------
    def display_current_channel(self):
        if self.stack is None:
            return
        if self.stack.ndim == 4:
            ch_img = self.stack[self.z_index, self.current_channel]
        else:
            ch_img = self.stack[self.current_channel]
        self.image_view.setImage(ch_img, autoLevels=True)   # transpose: pg uses (X,Y)

    def update_display(self):
        # show single channel image or composite image based on self.show_composite_cb state
        if self.show_composite_cb.isChecked():
            self.show_composite(True)
        else:
            self.display_current_channel()

    def update_scalebar(self):
        """Update the scalebar visibility and length."""
        # get the viewbox from the image view
        if self.scalebar:
            self.scalebar.remove()
            self.scalebar = None
            del self.scalebar  # clean up reference

        if self.scalebar_visible:
            vb: pg.ViewBox = self.image_view.view.getViewBox()
            # show the scalebar on the image view
            px_size = self.fov_w_edit.value() / self.stack.shape[2]  # µm / px
            self.scalebar = ScaleBar(vb, px_size, scale_bar_size=self.scalebar_length_um, unit='µm')



    def update_profile(self):
        """Re‑compute intensity profile for all channels along the ROI line."""
        if self.stack is None:
            return

        # 1. Get ROI endpoints in image coordinates (pixels)
        handles = self.roi.getSceneHandlePositions()
        if len(handles) != 2:
            return
        # the object returned by getView() can be a ViewBox *or* a PlotItem
        view_obj = self.image_view.getView()
        view_box = view_obj.getViewBox() if hasattr(view_obj, "getViewBox") else view_obj

        p0 = view_box.mapSceneToView(handles[0][1])
        p1 = view_box.mapSceneToView(handles[1][1])

        x0, x1 = p0.x(), p1.x()
        y0, y1 = p0.y(), p1.y()

        """
        # convert to pixel indices --------------------------------------------------
        px0 = self.image_view.x_mm_to_px_x(p0.x()) if hasattr(self.image_view, "x_mm_to_px_x") else x0
        py0 = self.image_view.y_mm_to_px_y(p0.y()) if hasattr(self.image_view, "y_mm_to_px_y") else y0
        px1 = self.image_view.x_mm_to_px_x(p1.x()) if hasattr(self.image_view, "x_mm_to_px_x") else x1
        py1 = self.image_view.y_mm_to_px_y(p1.y()) if hasattr(self.image_view, "y_mm_to_px_y") else y1
        self.profile_canvas.ax.set_xlabel("Distance (px)")

        profiles = []
        for ch in range(self.stack.shape[0]):
            prof = profile_line(
                self.stack[ch], (py0, px0), (py1, px1), mode="reflect", reduce_func=None
            )
            profiles.append(prof)
        """
        profiles = []

        # reopen the array exactly as used for display ---------------------------
        if self.stack.ndim == 4:
            view_arrs = [self.stack[self.z_index, c].T for c in range(self.stack.shape[1])]
        else:
            view_arrs = [ch.T for ch in self.stack]
        # reference to the ImageItem you passed those arrays to
        img_item = self.image_view.imageItem

        for arr in view_arrs:
            # For a LineSegmentROI the returned array has shape (1, N)
            line_img, coords = self.roi.getArrayRegion(
                arr,  # (X,Y) data as shown
                img_item,  # the same ImageItem
                axes=(0, 1),  # axis 0 -> X, axis 1 -> Y
                returnMappedCoords=True
            )
            profiles.append(line_img.squeeze())  # -> 1-D intensity profil
        # calculate the r coordinates from 0 to the length of the profile

        # get the pixel distance from the first point to all other points
        r0 = np.array([coords[0, 0], coords[1, 0]])
        r = np.sqrt((coords[0] - r0[0]) ** 2 + (coords[1] - r0[1]) ** 2)

        # convert to physical units if needed
        if self.show_unit_cb.isChecked():
            r = self.image_view.px_x_to_x_mm(r)
            r -= r[0]  # start at 0

            # alternatively use x0, x1, y0 and y1 to calculate the distance
            # r = np.linspace(0, np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), len(profiles[0]))

        self.profile_canvas.plot_profiles(profiles, coords=r)

    def normalize_channels(self, img: np.ndarray) -> np.ndarray:
        """ Normalize all channel data to the maximum of the data set."""
        if self.stack is None:
            return img

        if self.normalize_cb.isChecked():
            # normalize each channel independently
            max_vals = np.max(self.stack, axis=(1,2), keepdims=True)
            normalized = self.stack / max_vals
            self.stack = normalized.astype(np.float32)
        else:
            # restore original stack
            self.stack = self.stack_original.copy()

        self.update_display()
        self.update_profile()  # re-compute profile with normalized data

    # ------------------------------------------------------------------
    def correlate_channels(self, state: bool = True):
        """
        Align every channel to channel‑0 using FFT phase correlation.

        * Works with 3‑D stacks  (C, Y, X)
        * and with 4‑D stacks   (Z, C, Y, X)
        * Sub‑pixel shifts via scipy fourier_shift
        """
        if self.stack is None:
            return

        if not state:                       # restore original
            if self.stack_original is not None:
                self.stack = self.stack_original.copy()
                self.update_display()
                self.update_profile()
            return

        # ----- preserve original -------------------------------------
        self.stack_original = self.stack.copy()

        if self.stack.ndim == 3:            # -------- 3‑D (C,Y,X) -----
            ref   = self.stack[0]
            aligned = [ref]
            for ch in range(1, self.stack.shape[0]):
                moving = self.stack[ch]
                lo, hi = moving.min(), moving.max()
                shift, *_ = phase_cross_correlation(ref, moving, upsample_factor=10)
                moved = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(moving), shift)))
                aligned.append(np.clip(moved, lo, hi))
            self.stack = np.stack(aligned, axis=0)

        else:                               # -------- 4‑D (Z,C,Y,X) ---
            z_aligned = []
            n_ch = self.stack.shape[1]
            for z in range(self.z_size):
                ref = self.stack[z, 0]
                aligned_ch = [ref]
                for ch in range(1, n_ch):
                    moving = self.stack[z, ch]
                    lo, hi = moving.min(), moving.max()
                    shift, *_ = phase_cross_correlation(ref, moving, upsample_factor=10)
                    moved = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(moving), shift)))
                    aligned_ch.append(np.clip(moved, lo, hi))
                z_aligned.append(np.stack(aligned_ch, axis=0))
            self.stack = np.stack(z_aligned, axis=0)

        # ----- refresh view ------------------------------------------
        self.update_display()
        self.update_profile()

    def change_channel_labels(self):
        """
        Change the labels of the channels in the profile canvas.
        Opens a dialog to enter new labels for each channel.
        """
        if self.stack is None:
            return

        num_channels = self.stack.shape[0]
        # make a grid layout with a label and a text input for each channel
        labels = self.profile_canvas.labels if self.profile_canvas.labels else [f"Channel {i}" for i in range(num_channels)]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Change Channel Labels")
        layout = QtWidgets.QGridLayout(dialog)
        for i in range(num_channels):
            label = QtWidgets.QLabel(f"Channel {i}:")
            text_input = QtWidgets.QLineEdit(labels[i])
            layout.addWidget(label, i, 0)
            layout.addWidget(text_input, i, 1)
        # add OK and Cancel buttons
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons, num_channels, 0, 1, 2)
        # show the dialog
        ok = dialog.exec()
        if not ok:
            return
        # get the labels from the text inputs
        labels = []
        for i in range(num_channels):
            text_input = layout.itemAtPosition(i, 1).widget()
            if isinstance(text_input, QtWidgets.QLineEdit):
                labels.append(text_input.text())
        # check if the number of labels matches the number of channels
        ok = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
        # if ok, update the labels in the profile canvas
        # and update the profile
        # if the labels are not empty and match the number of channels
        if not ok:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please enter labels for all channels."
            )
            return
        # check if the labels are not empty and match the number of channels
        if len(labels) != num_channels or not all(labels):
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please enter labels for all channels."
            )
            return
        self.profile_canvas.labels = labels
        # update the profile canvas labels
        self.profile_canvas.ax.legend(self.profile_canvas.labels)
        self.profile_canvas.draw_idle()


    def open_channel_colour_dialog(self):
        dlg = ChannelColorDialog(self.profile_canvas.colours, self)
        result = dlg.exec()
        try:
            accepted_code = QtWidgets.QDialog.DialogCode.Accepted  # PyQt6 / PySide6
        except AttributeError:
            accepted_code = QtWidgets.QDialog.Accepted             # PyQt5 / PySide2

        if result == accepted_code:
            # update colours
            self.profile_canvas.colours = dlg.colours()
            # trigger plot and image refresh
            if self.stack is not None:
                self.profile_canvas.ax.legend(self.profile_canvas.labels or
                                              [f"Ch {i}" for i in range(self.stack.shape[0])],
                                              prop={'size': 10})
                self.update_display()
                self.update_profile()

    def update_units(self, state: int):
        self.profile_canvas.ax.set_xlabel("Distance (µm)" if state else "Distance (px)")



# --------------------------------------------------------------------------------------
#  Main entry point
# --------------------------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')  # use y,x for images
    win = LineScanApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
