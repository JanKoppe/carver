# Seam carving implementation

This is a rudimentary implementation of the seam carving algorithm described by the paper "Avidan, S. et al. (2007) Seam Carving For Content–Aware Image Resizing. ACM Transactions on Graphics (TOG),
26(3)", which was a assignment for the "Computer Vision" class 2016/2017 taught by Prof. Dr. Xiaoyi Jiang at the University of Münster, Germany.

The original paper and explanation for the seam carving algorithm can be found here: [http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html).

OpenCV has only been used to load/save the image and do very rudimentary pixel operations, as this was a restriction in the original assignment.

# Requirements

```
opencv>=2.4
```

The project uses `qmake` and thus `Qt>=4` for Makefile generation.

# Building

```
qmake
make
```

# Usage

```
$ carver
  Usage: ./carver input-image new-x-dimension new-y-dimension output-image
```

Carver takes the filename of a input image, the target dimensions and the filename for the output image.

Example:
```
carver input.jpg 50 50 output.jpg
```

This will load the `input.jpg` image, remove as many seams using the seam carving algorithm as necessary to reach a 50x50 image size, and save it as `output.jpg`.

# Examples

Taken from https://commons.wikimedia.org/wiki/Commons:Quality_images/de#/media/File:Chess_pawn.jpg, licensed as CC-BA-SA-4.0, taken by WikiMedia user "Svklimkin".

Before: Original size 640 by 438 pixels.

![example_before.jpg](example_before.jpg)

After: Target size 400 by 400 pixels.

![example_after.jpg](example_after.jpg)

# Known Issues

For some reason, this program sometimes fails when removing a lot of seams, depending on the input image. This seems to come from the `.copyTo()` in the main loop, which will move the work copies around on each seam iteration. This could probably be avoided with some clever pointer logic, but I'm not proficient enough in C++ to do so.

Also, the last row/col stays black in the `removeSeam()` method. Something is not quite right there yet.

# Contributing

If you want to contribute, feel free to do so on [github.com/JanKoppe/carver](https://github.com/JanKoppe/carver). Please do note though that this was a one-off assignment that I will most likely not work on again. The repository is just for archival purposes.

# License

MIT License. See attached `LICENSE` file.