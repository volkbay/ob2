# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_OB2(object):
    def setupUi(self, OB2):
        if not OB2.objectName():
            OB2.setObjectName(u"OB2")
        OB2.resize(800, 600)
        self.centralwidget = QWidget(OB2)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.pushButton_2)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setIconSize(QSize(16, 16))

        self.horizontalLayout_2.addWidget(self.pushButton)

        OB2.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(OB2)
        self.statusbar.setObjectName(u"statusbar")
        OB2.setStatusBar(self.statusbar)

        self.retranslateUi(OB2)

        QMetaObject.connectSlotsByName(OB2)
    # setupUi

    def retranslateUi(self, OB2):
        OB2.setWindowTitle(QCoreApplication.translate("OB2", u"OB2.0", None))
    # retranslateUi

