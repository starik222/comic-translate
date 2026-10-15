from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.spin_box import MSpinBox
from .utils import create_title_and_combo, set_combo_box_width
from modules.utils.device import is_gpu_available, get_available_gpus

class ToolsPage(QtWidgets.QWidget):
    def __init__(
        self, 
        translators: list[str], 
        ocr_engines: list[str], 
        detectors: list[str],
        inpainters: list[str], 
        inpaint_strategy: list[str], 
        parent=None
    ):
        super().__init__(parent)
        self.translators = translators
        self.ocr_engines = ocr_engines
        self.detectors = detectors
        self.inpainters = inpainters
        self.inpaint_strategy = inpaint_strategy

        layout = QtWidgets.QVBoxLayout(self)

        translator_widget, self.translator_combo = create_title_and_combo(self.tr("Translator"), self.translators, h4=True)
        set_combo_box_width(self.translator_combo, self.translators)

        ocr_widget, self.ocr_combo = create_title_and_combo(self.tr("Text Recognition"), self.ocr_engines, h4=True)
        set_combo_box_width(self.ocr_combo, self.ocr_engines)

        detector_widget, self.detector_combo = create_title_and_combo(self.tr("Text Detector"), self.detectors, h4=True)
        set_combo_box_width(self.detector_combo, self.detectors)

        inpainting_label = MLabel(self.tr("Image Cleaning")).h4()
        inpainter_widget, self.inpainter_combo = create_title_and_combo(self.tr("Inpainter"), self.inpainters, h4=False)
        set_combo_box_width(self.inpainter_combo, self.inpainters)
        self.inpainter_combo.setCurrentText(self.tr("AOT"))

        inpaint_strategy_widget, self.inpaint_strategy_combo = create_title_and_combo(self.tr("HD Strategy"), self.inpaint_strategy, h4=False)
        set_combo_box_width(self.inpaint_strategy_combo, self.inpaint_strategy)
        self.inpaint_strategy_combo.setCurrentText(self.tr("Resize"))

        # HD Strategy detail widgets
        self.hd_strategy_widgets = QtWidgets.QWidget()
        self.hd_strategy_layout = QtWidgets.QVBoxLayout(self.hd_strategy_widgets)

        # Resize panel
        self.resize_widget = QtWidgets.QWidget()
        about_resize_layout = QtWidgets.QVBoxLayout(self.resize_widget)
        resize_layout = QtWidgets.QHBoxLayout()
        resize_label = MLabel(self.tr("Resize Limit:"))
        about_resize_label = MLabel(self.tr("Resize the longer side of the image to a specific size,\nthen do inpainting on the resized image."))
        self.resize_spinbox = MSpinBox().small()
        self.resize_spinbox.setFixedWidth(70)
        self.resize_spinbox.setMaximum(3000)
        self.resize_spinbox.setValue(960)
        resize_layout.addWidget(resize_label)
        resize_layout.addWidget(self.resize_spinbox)
        resize_layout.addStretch()
        about_resize_layout.addWidget(about_resize_label)
        about_resize_layout.addLayout(resize_layout)
        about_resize_layout.setContentsMargins(5, 5, 5, 5)
        about_resize_layout.addStretch()

        # Crop panel
        self.crop_widget = QtWidgets.QWidget()
        crop_layout = QtWidgets.QVBoxLayout(self.crop_widget)
        about_crop_label = MLabel(self.tr("Crop masking area from the original image to do inpainting."))
        crop_margin_layout = QtWidgets.QHBoxLayout()
        crop_margin_label = MLabel(self.tr("Crop Margin:"))
        self.crop_margin_spinbox = MSpinBox().small()
        self.crop_margin_spinbox.setFixedWidth(70)
        self.crop_margin_spinbox.setMaximum(3000)
        self.crop_margin_spinbox.setValue(512)
        crop_margin_layout.addWidget(crop_margin_label)
        crop_margin_layout.addWidget(self.crop_margin_spinbox)
        crop_margin_layout.addStretch()

        crop_trigger_layout = QtWidgets.QHBoxLayout()
        crop_trigger_label = MLabel(self.tr("Crop Trigger Size:"))
        self.crop_trigger_spinbox = MSpinBox().small()
        self.crop_trigger_spinbox.setFixedWidth(70)
        self.crop_trigger_spinbox.setMaximum(3000)
        self.crop_trigger_spinbox.setValue(512)
        crop_trigger_layout.addWidget(crop_trigger_label)
        crop_trigger_layout.addWidget(self.crop_trigger_spinbox)
        crop_trigger_layout.addStretch()

        crop_layout.addWidget(about_crop_label)
        crop_layout.addLayout(crop_margin_layout)
        crop_layout.addLayout(crop_trigger_layout)
        crop_layout.setContentsMargins(5, 5, 5, 5)

        self.hd_strategy_layout.addWidget(self.resize_widget)
        self.hd_strategy_layout.addWidget(self.crop_widget)

        self.resize_widget.show()
        self.crop_widget.hide()
        self.inpaint_strategy_combo.currentIndexChanged.connect(self._update_hd_strategy_widgets)


        self.use_gpu_checkbox = MCheckBox(self.tr("Use GPU"))
        self.available_gpus = get_available_gpus()
        gpu_options = [f"GPU {g['id']}: {g['name']}" for g in self.available_gpus]
        if not gpu_options:
            gpu_options = ["GPU 0"]

        self.gpu_widget, self.gpu_device_combo = create_title_and_combo(
            self.tr("GPU Device"),
            gpu_options,
            h4=False
        )
        for i, g in enumerate(self.available_gpus):
            self.gpu_device_combo.setItemData(i, g['id'])
        set_combo_box_width(self.gpu_device_combo, gpu_options)

        gpu_available = is_gpu_available()
        if not gpu_available:
            self.use_gpu_checkbox.setVisible(False)
            self.gpu_widget.setVisible(False)
        else:
            self.use_gpu_checkbox.setVisible(True)
            self._update_gpu_widget_visibility()
            self.use_gpu_checkbox.stateChanged.connect(self._update_gpu_widget_visibility)


        layout.addWidget(translator_widget)
        layout.addSpacing(10)
        layout.addWidget(detector_widget)
        layout.addSpacing(10)
        layout.addWidget(ocr_widget)
        layout.addSpacing(10)
        layout.addWidget(inpainting_label)
        layout.addWidget(inpainter_widget)
        layout.addWidget(inpaint_strategy_widget)
        layout.addWidget(self.hd_strategy_widgets)
        layout.addSpacing(10)
        layout.addWidget(self.use_gpu_checkbox)
        layout.addWidget(self.gpu_widget)
        layout.addStretch(1)

        self._update_hd_strategy_widgets(self.inpaint_strategy_combo.currentIndex())

    def _update_gpu_widget_visibility(self):
        is_checked = self.use_gpu_checkbox.isChecked()
        self.gpu_widget.setVisible(is_checked and len(self.available_gpus) > 0)

    def _update_hd_strategy_widgets(self, index: int):
        strategy = self.inpaint_strategy_combo.itemText(index)
        self.resize_widget.setVisible(strategy == self.tr("Resize"))
        self.crop_widget.setVisible(strategy == self.tr("Crop"))
        if strategy == self.tr("Original"):
            self.hd_strategy_widgets.setFixedHeight(0)
        else:
            self.hd_strategy_widgets.setFixedHeight(self.hd_strategy_widgets.sizeHint().height())
