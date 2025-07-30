import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
import logging

from PyQt6.QtCore import QPointF
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

class ImageProcessingThread(QtCore.QThread):
    """Thread for processing images in the background."""
    finished = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, image, aspect_ratio, axis_order, interpol_order=3):
        super().__init__()
        self.image = image
        self.aspect_ratio = aspect_ratio
        self.axis_order = axis_order
        self.interpol_order = interpol_order

    def update(self, image, **kwargs):
        # kwargs are attributes to be set on the thread object
        self.image = image
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        if self.image is None:
            raise ValueError("No image to process")

        im = self.image.copy()
        img_aspect = im.shape[self.axis_order['x']] / im.shape[self.axis_order['y']]
        logger.debug(f"Image Processing: {img_aspect=}; {self.aspect_ratio=}")
        if img_aspect < self.aspect_ratio:
            scale_factor = self.aspect_ratio / img_aspect
            im = zoom(im, (1, scale_factor), order=self.interpol_order)
            logger.debug(f"Image Processing: Scaling image by {scale_factor}")
        elif img_aspect > self.aspect_ratio:
            scale_factor = img_aspect / self.aspect_ratio
            im = zoom(im, (scale_factor, 1), order=self.interpol_order)
            logger.debug(f"Image Processing: Scaling image by {scale_factor}")
        else:
            logger.debug("Image Processing: No scaling needed")
        self.finished.emit(im)  # Emit the processed image

class PhysicalUnitsImageView(pg.ImageView):
    """Custom ImageView class that handles physical units. Only accepts 2D images."""
    # FIXME: be careful. The ImageItem.image axes are swapped compared to the input image
    def __init__(self, width, height, aspect_ratio=None,
                 physical_unit='mm', axes={'x': 1, 'y': 0},
                 rescale=True):
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.plot_item_units = pg.PlotItem()
        super().__init__(view=self.plot_item_units)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.view.setDefaultPadding(0)
        self.setColorMap(pg.colormap.getFromMatplotlib('viridis'))

        self.physical_unit = physical_unit
        self.width = width
        self.height = height
        self.axis_order = axes
        self.transform = QtGui.QTransform()
        self.aspect_ratio = aspect_ratio if aspect_ratio else self.width / self.height
        # self.n_px_x = self.width / self.pixel_size_x
        # self.n_px_y = self.height / self.pixel_size_y
        self.rescale = rescale
        self.levels_set = False  # Track whether levels have been set
        # Set labels for axes with the physical units
        self.plot_item_units.setLabel('bottom', '', units=self.physical_unit)
        self.plot_item_units.setLabel('left', '', units=self.physical_unit)

        # Set the range of the axes so that 0 is at the center
        self.plot_item_units.setXRange(-width / 2, width / 2)
        self.plot_item_units.setYRange(-height / 2, height / 2)

        self.processing_thread = None
        # For handling the background thread
        if self.rescale:
            self.processing_thread = ImageProcessingThread(None, self.aspect_ratio, self.axis_order, interpol_order=3)
            self.processing_thread.finished.connect(self.on_image_ready)

    def adjust_dimensions(self, width, height, unit=None, aspect_ratio=None):
        if unit is not None:
            self.physical_unit = unit
            self.plot_item_units.setLabel('bottom', '', units=self.physical_unit)
            self.plot_item_units.setLabel('left', '', units=self.physical_unit)
        self.width = width
        self.height = height
        self.aspect_ratio = width / height if aspect_ratio is None else aspect_ratio
        self.plot_item_units.setXRange(-width / 2, width / 2)
        self.plot_item_units.setYRange(-height / 2, height / 2)
        self.center_image_at_origin()


    def center_image_at_origin(self):
        # IMPORTANT: in image item the x axis is always axis 0!!!
        if self.imageItem.image is None:
            return
        self.pixel_size_x = self.width / self.imageItem.image.shape[0]
        self.pixel_size_y = self.height / self.imageItem.image.shape[1]
        self.transform.reset()
        # add half a pixel size to have center of the center pixel at 0,0
        self.transform.translate(-self.width / 2 - self.pixel_size_x/2, -self.height / 2 - self.pixel_size_y/2)
        self.transform.scale(self.width / self.imageItem.image.shape[0], self.height / self.imageItem.image.shape[1])
        self.imageItem.setTransform(self.transform)

    def px_x_to_x_mm(self, px_x: float) -> float:
        if self.imageItem.image is None:
            return None
        # Convert pixel x to physical x
        # Apply the translation and scale
        translate = self.width / 2 + self.pixel_size_x/2
        scale = self.width / self.imageItem.image.shape[0]
        x_mm = px_x * scale - translate
        return x_mm

    def x_mm_to_px_x(self, x: float) -> float:
        if self.imageItem.image is None:
            return None
        # Convert physical x to pixel x
        # Reverse the translation and scale
        back_translate = - (-self.width / 2 - self.pixel_size_x/2)
        back_scale = 1 / (self.width / self.imageItem.image.shape[0])
        px_x = (x + back_translate)*back_scale
        return px_x

    def y_mm_to_px_y(self, y):
        # Convert physical y to pixel y
        # Reverse the translation and scale
        if self.imageItem.image is None:
            return None
        back_translate = - (-self.height / 2 - self.pixel_size_y/2)
        back_scale = 1 / (self.height / self.imageItem.image.shape[1])
        px_y = (y + back_translate) * back_scale
        return px_y

    def length_x_to_px_x(self, width_mm):
        if self.imageItem.image is None:
            return None
        # Convert a physical length to pixels
        return round(width_mm / self.pixel_size_x)

    def length_y_to_px_y(self, height_mm):
        if self.imageItem.image is None:
            return None
        # Convert a physical length to pixels
        return round(height_mm / self.pixel_size_y)

    def setImage(self, *args, rescale: bool = None, interpol_order=3, autoHistogramRange=False, autoLevels=False,
                 **kwargs):
        if self.processing_thread is not None:
            if self.processing_thread.isRunning():
                logger.debug('Image Processor: Still processing image')
                return
        im = args[0]
        if rescale is None:
            rescale = self.rescale

        if 'axes' in kwargs:
            kwargs.pop('axes')

        if rescale:
            #image_setter = lambda x: super(PhysicalUnitsImageView, self).setImage(x, *args[1:], axes=self.axis_order,
            #                                                                      **kwargs)
            # TODO: handle 3D images
            self.processing_thread.update(im)
            self.processing_thread.start()
        else:
            super().setImage(im, *args[1:], axes=self.axis_order, **kwargs, autoHistogramRange=autoHistogramRange,
                             autoLevels=autoLevels)
            self.center_image_at_origin()

            # Set the levels on the first image
            if not self.levels_set:
                hist = self.getHistogramWidget()
                if hist is not None:
                    vmin, vmax = np.nanmin(im), np.nanmax(im)
                    hist.vb.setLimits(yMin=None, yMax=None)
                    self.setLevels(vmin, vmax)
                    self.levels_set = True
            # Restore margins and scale
            self.plot_item_units.setXRange(-self.width / 2, self.width / 2)
            self.plot_item_units.setYRange(-self.height / 2, self.height / 2)

    def add_marker(self, x_physical, y_physical, marker_size=8, color=(255, 0, 0)):
        """Add a marker at a specific (x, y) position in mm."""
        marker = pg.ScatterPlotItem(
            pos=np.array([[x_physical, y_physical]]),
            size=marker_size,
            brush=color,
            pen=pg.mkPen(color, width=1),  # Outline color and width
            symbol='x',  # Change this to 'x' for a different cross
        )
        self.getView().addItem(marker)
        return marker

    def add_box(self, x_physical, y_physical, width_1_mm: tuple or int, color=(255, 0, 0)):
        """Add a box at a specific (x, y) position in mm."""

        # Box size (adjust this based on your data units)
        pen = pg.mkPen(color='r', width=2)

        if isinstance(width_1_mm, (tuple, list)):
            width_1_mm_y, width_1_mm_x = width_1_mm
        else:
            width_1_mm_y = width_1_mm_x = width_1_mm
        # Create red boxes as ROIs or QGraphicsRectItem
        half_width_x = width_1_mm_x / 2
        half_width_y = width_1_mm_y / 2
        order_box1 = LockROI([x_physical - half_width_x, y_physical - half_width_y], [width_1_mm_x, width_1_mm_y], pen=pen,
                                     movable=False)
        order_box2 = LockROI([-x_physical - half_width_x, -y_physical - half_width_y], [width_1_mm_x, width_1_mm_y], pen=pen,
                                     movable=False)

        order_box1.lock_movement()
        order_box2.lock_movement()

        order_box1.scale_changed.connect(lambda box: order_box2.sync_boxes(box))
        order_box2.scale_changed.connect(lambda box: order_box1.sync_boxes(box))

        # Add them to the image view's ViewBox
        self.view.addItem(order_box1)
        self.view.addItem(order_box2)

        return order_box1, order_box2

    def remove_marker(self, marker):
        view = self.getView()
        view.removeItem(marker)

    def remove_box(self, box):
        view = self.getView()
        view.removeItem(box)

    def setImage(self, *args, **kwargs):
        view = self.getView()
        view_range = view.viewRange()
        # ignore the axes argument
        if 'axes' in kwargs:
            kwargs.pop('axes')
        # ignore autoHistogramRange and autoLevels
        # Update the image
        super().setImage(*args, axes=self.axis_order, **kwargs)
        # Restore the saved view range
        view.setXRange(*view_range[0], padding=0)
        view.setYRange(*view_range[1], padding=0)

        # Center image at origin
        self.center_image_at_origin()

    @QtCore.pyqtSlot(np.ndarray)
    def on_image_ready(self, processed_image):
        # Update the image
        self.setImage(processed_image)

        if not self.levels_set:
            logger.debug('setting levels')
            hist = self.getHistogramWidget()
            if hist is not None:
                vmin, vmax = np.nanmin(processed_image), np.nanmax(processed_image)
                hist.vb.setLimits(yMin=None, yMax=None)
                self.setLevels(vmin, vmax)
                self.levels_set = True


class LockROI(pg.RectROI):
    scale_changed = QtCore.pyqtSignal(pg.ROI)
    def __init__(self, *args, lock_aspect=False, **kwargs):
        self.centered_resizing_enabled = False
        super().__init__(*args, **kwargs)
        roi = self # The ROI to be locked
        self.roi = roi
        self.fill_item = None  # Overlay fill item
        self._original_translate = roi.translate  # Save original translate method
        self._original_setSize = roi.setSize if hasattr(roi, "setSize") else None  # Save resizing method
        # Connect signals to update fill dynamically
        self.sigRegionChanged.connect(self.reapply_fill)
        self.centered_resizing_enabled = True
        # Shared flag to avoid recursive updates
        self._resizing_sync_guard = False
        self.lock_aspect = lock_aspect

    def physical_size_1_um(self):
        """Return the physical size of the ROI in inverse micrometers."""
        if self.roi is None or self.roi.size() is None:
            return None
        size = self.roi.size()
        print("ROI size in 1/mm: (h, w)", size.y(), size.x())
        return size.x() * 1e-3, size.y() * 1e-3

    def physical_size_1_mm(self):
        """Return the physical size of the ROI in millimeters."""
        if self.roi is None or self.roi.size() is None:
            return None
        size = self.roi.size()
        print("ROI size in 1/mm: (h, w)", size.y(), size.x())
        return size.x(), size.y()

    def sync_boxes(self, other_box: pg.ROI):
        if self._resizing_sync_guard:
            return

        try:
            self._resizing_sync_guard = True

            # Get new size from other box
            new_size = other_box.size()

            # Compute self's center (before resizing)
            center = self.pos() + self.size() * 0.5

            # Compute new position that keeps center fixed
            new_pos = center - new_size * 0.5

            self.blockSignals(True)
            self.setSize([new_size.x(), new_size.y()])
            self.setPos(new_pos)
            self.blockSignals(False)

        finally:
            self._resizing_sync_guard = False

    def setSize(self, size, **kwargs):
        # FIXME: lock and unlock aspect ration
        """Override setSize to keep center fixed and preserve aspect ratio during resize."""
        if self.centered_resizing_enabled:
            # Preserve aspect ratio using the original ROI size
            current_size = self.size()
            current_width, current_height = current_size.x(), current_size.y()

            # Desired new size
            new_width, new_height = size[0], size[1]

            # Compute scaling factors
            scale_w = new_width / current_width if current_width else 1
            scale_h = new_height / current_height if current_height else 1

            if self.lock_aspect:
                # If lock_aspect is True, maintain aspect ratio
                if scale_w == 0 or scale_h == 0:
                    return
                # Use the smaller scale to maintain aspect ratio
                scale = min(scale_w, scale_h)
                scale_w = scale_h = scale

            aspect_width = current_width * scale_w
            aspect_height = current_height * scale_h

            center = self.pos() + current_size * 0.5
            new_pos = center - QPointF(aspect_width, aspect_height) * 0.5

            super().setSize([aspect_width, aspect_height], **kwargs)
            self.setPos(new_pos)
            self.scale_changed.emit(self.roi)  # Emit signal for scaling change
        else:
            super().setSize(size, **kwargs)

    def _enforce_center_resize(self):
        """Called on ROI resize; keeps center fixed."""
        new_size = self.size()
        new_center = self.pos() + self.size() * 0.5
        new_pos = new_center - new_size * 0.5
        self.blockSignals(True)
        self.setPos(new_pos)
        self.blockSignals(False)

    def fill(self):
        """Fill the ROI with the same color as its pen color."""
        self.clear_fill()  # Ensure previous fill is removed

        # Create a new fill item based on the ROI's shape
        path = self.shape()  # Get the exact shape
        self.fill_item = QtWidgets.QGraphicsPathItem(path)

        color = self.pen.color()  # Get the ROI's pen color
        color.setAlpha(100)  # Set fill color to 100/255 transparency
        self.fill_item.setBrush(pg.mkBrush(color.lighter(150)))  # Lighter shade of ROI color
        self.fill_item.setPen(pg.mkPen(None))  # No border

        # Attach fill item to ROI so it moves/resizes together
        self.fill_item.setParentItem(self)
        # Optional: lock features of the ROI
        # self.lock_resizing()
        # self.lock_movement()

    def clear_fill(self):
        """Remove the existing fill item."""
        if self.fill_item is not None:
            self.fill_item.setParentItem(None)  # Detach from ROI
            self.fill_item.setPen(pg.mkPen(None))  # Ensure no border
            self.fill_item.setBrush(pg.mkBrush(None))  # Ensure no fill
            self.fill_item = None  # Reset reference


    def reapply_fill(self):
        """Reapply fill dynamically when ROI is resized or moved."""
        if self.fill_item is None:
            return  # No fill to update
        self.fill()  # Automatically clears old fill and applies new one

    def lock_movement(self):
        """Disable movement."""
        self.translate = lambda *args, **kwargs: None

    def unlock_movement(self):
        """Enable movement."""
        if hasattr(self, "_original_translate"):
            self.translate = self._original_translate

    def lock_resizing(self):
        """Disable resizing if the ROI supports setSize()."""
        if self._original_setSize is not None:
            self.setSize = lambda *args, **kwargs: None

    def unlock_resizing(self):
        """Enable resizing."""
        if self._original_setSize is not None:
            self.setSize = self._original_setSize

    def restore_original_roi(self):
        """Restore original translate and setSize methods of the ROI."""
        self.unlock_movement()
        self.unlock_resizing()  # Unlock resizing if supported
        self.clear_fill()