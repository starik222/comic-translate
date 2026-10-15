from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.text_edit import MTextEdit
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.collapse import MCollapse
from ..dayu_widgets.spin_box import MSpinBox
from ..dayu_widgets.push_button import MPushButton

class LlmsPage(QtWidgets.QWidget):
    DEFAULT_EXTRA_CONTEXT_LIMIT = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._extra_context_limit: int | None = self.DEFAULT_EXTRA_CONTEXT_LIMIT

        v = QtWidgets.QVBoxLayout(self)
        main_layout = QtWidgets.QHBoxLayout()

        self.image_checkbox = MCheckBox()
        self.set_image_context_credits(1)
        self.image_checkbox.setChecked(False)

        # Context settings
        self.context_checkbox = MCheckBox(self.tr("Use translation context (include previous requests/responses)"))
        self.context_checkbox.setChecked(False)

        context_pair_layout = QtWidgets.QHBoxLayout()
        self.context_label = MLabel(self.tr("Previous request/response pairs:"))
        self.context_limit_spinbox = MSpinBox().small()
        self.context_limit_spinbox.setMinimum(1)
        self.context_limit_spinbox.setMaximum(50)
        self.context_limit_spinbox.setValue(3)
        self.context_limit_spinbox.setFixedWidth(70)

        self.clear_context_button = MPushButton(self.tr("Clear Context")).small()
        self.clear_context_button.setFixedWidth(110)

        context_pair_layout.addWidget(self.context_label)
        context_pair_layout.addWidget(self.context_limit_spinbox)
        context_pair_layout.addSpacing(15)
        context_pair_layout.addWidget(self.clear_context_button)
        context_pair_layout.addStretch(1)

        # Left
        left_layout = QtWidgets.QVBoxLayout()
        prompt_label = MLabel(self.tr("Extra Context:"))
        self.extra_context = MTextEdit()
        self.extra_context.setMinimumHeight(200)
        left_layout.addWidget(prompt_label)
        left_layout.addWidget(self.extra_context)
        left_layout.addWidget(self.image_checkbox)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.context_checkbox)
        left_layout.addLayout(context_pair_layout)
        left_layout.addStretch(1)

        # Right
        right_layout = QtWidgets.QVBoxLayout()

        # Advanced settings

        right_layout.addSpacing(10)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        v.addLayout(main_layout)
        v.addStretch(1)

        self.extra_context.textChanged.connect(self._limit_extra_context)
        self.context_checkbox.toggled.connect(self._on_context_checkbox_toggled)
        self._on_context_checkbox_toggled(self.context_checkbox.isChecked())

    def _on_context_checkbox_toggled(self, checked: bool) -> None:
        self.context_label.setEnabled(checked)
        self.context_limit_spinbox.setEnabled(checked)
        self.clear_context_button.setEnabled(checked)

    def set_extra_context_unlimited(self, enabled: bool) -> None:
        self._extra_context_limit = None if enabled else self.DEFAULT_EXTRA_CONTEXT_LIMIT
        self._limit_extra_context()

    def set_image_context_credits(self, credits: int) -> None:
        unit = "Credit" if credits == 1 else "Credits"
        self.image_checkbox.setText(f"{self.tr('Provide Image as Input to AI')} (+{credits} {unit})")

    def _limit_extra_context(self):
        max_length = self._extra_context_limit
        if max_length is None:
            return
        text = self.extra_context.toPlainText()
        if len(text) > max_length:
            # Preserve cursor position
            cursor = self.extra_context.textCursor()
            position = cursor.position()
            
            # Truncate
            self.extra_context.setPlainText(text[:max_length])
            
            # Restore cursor (clamped to end)
            new_position = min(position, max_length)
            cursor.setPosition(new_position)
            self.extra_context.setTextCursor(cursor)
