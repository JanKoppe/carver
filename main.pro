TARGET = carver
TEMPLATE = app

SOURCES +=main.cpp

unix {
  QMAKE_CXXFLAGS += -std=c++11 -Wall -pedantic -Wno-unknown-pragmas

  LIBS += -L/usr/local/lib \
          -lopencv_core \
          -lopencv_highgui \
          -lopencv_imgproc \
          -lopencv_imgcodecs

  QMAKE_CXXFLAGS_WARN_ON = -Wno-unused-variable -Wno-reorder
}
