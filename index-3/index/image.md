# 이미지 처리

Note: Functions taking `Tensor` arguments can also take anything accepted by [`tf.convert_to_tensor`](framework.md#convert\_to\_tensor).

\[TOC]

## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats. Encoded images are represented by scalar string Tensors, decoded images by 3-D uint8 tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

The encode and decode Ops apply to one image at a time. Their input and output are all of variable size. If you need fixed size images, pass the output of the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops presently only support RGB, HSV, and GrayScale. Presently, the alpha channel has to be stripped from the image and re-attached using slicing ops.

***

### `tf.image.decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None, try_recover_truncated=None, acceptable_fraction=None, name=None)` <a href="#decode_jpeg" id="decode_jpeg"></a>

Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the decoded image.

Accepted values are:

* 0: Use the number of channels in the JPEG-encoded image.
* 1: output a grayscale image.
* 3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number of color channels.

The attr `ratio` allows downscaling the image by an integer factor during decoding. Allowed values are: 1, 2, 4, and 8. This is much faster than downscaling the image later.

**Args:**

* `contents`: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
* `channels`: An optional `int`. Defaults to `0`. Number of color channels for the decoded image.
* `ratio`: An optional `int`. Defaults to `1`. Downscaling ratio.
* `fancy_upscaling`: An optional `bool`. Defaults to `True`. If true use a slower but nicer upscaling of the chroma planes (yuv420/422 only).
* `try_recover_truncated`: An optional `bool`. Defaults to `False`. If true try to recover an image from truncated input.
* `acceptable_fraction`: An optional `float`. Defaults to `1`. The minimum required fraction of lines before a truncated input is accepted.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`..

***

### `tf.image.encode_jpeg(image, format=None, quality=None, progressive=None, optimize_size=None, chroma_downsampling=None, density_unit=None, x_density=None, y_density=None, xmp_metadata=None, name=None)` <a href="#encode_jpeg" id="encode_jpeg"></a>

JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

The attr `format` can be used to override the color format of the encoded output. Values can be:

* `''`: Use a default format based on the number of channels in the image.
* `grayscale`: Output a grayscale JPEG image. The `channels` dimension of `image` must be 1.
* `rgb`: Output an RGB JPEG image. The `channels` dimension of `image` must be 3.

If `format` is not specified or is the empty string, a default format is picked in function of the number of channels in `image`:

* 1: Output a grayscale image.
* 3: Output an RGB image.

**Args:**

* `image`: A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`.
* `format`: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`. Per pixel image format.
* `quality`: An optional `int`. Defaults to `95`. Quality of the compression from 0 to 100 (higher is better and slower).
* `progressive`: An optional `bool`. Defaults to `False`. If True, create a JPEG that loads progressively (coarse to fine).
* `optimize_size`: An optional `bool`. Defaults to `False`. If True, spend CPU/RAM to reduce size with no quality change.
* `chroma_downsampling`: An optional `bool`. Defaults to `True`. See http://en.wikipedia.org/wiki/Chroma\_subsampling.
* `density_unit`: An optional `string` from: `"in", "cm"`. Defaults to `"in"`. Unit used to specify `x_density` and `y_density`: pixels per inch (`'in'`) or centimeter (`'cm'`).
* `x_density`: An optional `int`. Defaults to `300`. Horizontal pixels per density unit.
* `y_density`: An optional `int`. Defaults to `300`. Vertical pixels per density unit.
* `xmp_metadata`: An optional `string`. Defaults to `""`. If not empty, embed this XMP metadata in the image header.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `string`. 0-D. JPEG-encoded image.

***

### `tf.image.decode_png(contents, channels=None, dtype=None, name=None)` <a href="#decode_png" id="decode_png"></a>

Decode a PNG-encoded image to a uint8 or uint16 tensor.

The attr `channels` indicates the desired number of color channels for the decoded image.

Accepted values are:

* 0: Use the number of channels in the PNG-encoded image.
* 1: output a grayscale image.
* 3: output an RGB image.
* 4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number of color channels.

**Args:**

* `contents`: A `Tensor` of type `string`. 0-D. The PNG-encoded image.
* `channels`: An optional `int`. Defaults to `0`. Number of color channels for the decoded image.
* `dtype`: An optional `tf.DType` from: `tf.uint8, tf.uint16`. Defaults to `tf.uint8`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `dtype`. 3-D with shape `[height, width, channels]`.

***

### `tf.image.encode_png(image, compression=None, name=None)` <a href="#encode_png" id="encode_png"></a>

PNG-encode an image.

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]` where `channels` is:

* 1: for grayscale.
* 2: for grayscale + alpha.
* 3: for RGB.
* 4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder default or a value from 0 to 9. 9 is the highest compression level, generating the smallest output, but is slower.

**Args:**

* `image`: A `Tensor`. Must be one of the following types: `uint8`, `uint16`. 3-D with shape `[height, width, channels]`.
* `compression`: An optional `int`. Defaults to `-1`. Compression level.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `string`. 0-D. PNG-encoded image.

## Resizing

The resizing Ops accept input images as tensors of several types. They always output resized images as float32 tensors.

The convenience function [`resize_images()`](image.md#resize\_images) supports both 4-D and 3-D tensors as input and output. 4-D tensors are for batches of images, 3-D tensors for individual images.

Other resizing Ops only support 4-D batches of images as input: [`resize_area`](image.md#resize\_area), [`resize_bicubic`](image.md#resize\_bicubic), [`resize_bilinear`](image.md#resize\_bilinear), [`resize_nearest_neighbor`](image.md#resize\_nearest\_neighbor).

Example:

```python
# Decode a JPG image and resize it to 299 by 299 using default method.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_images(image, 299, 299)
```

***

### `tf.image.resize_images(images, new_height, new_width, method=0, align_corners=False)` <a href="#resize_images" id="resize_images"></a>

Resize `images` to `new_width`, `new_height` using the specified `method`.

Resized images will be distorted if their original aspect ratio is not the same as `new_width`, `new_height`. To avoid distortions see [`resize_image_with_crop_or_pad`](image.md#resize\_image\_with\_crop\_or\_pad).

`method` can be one of:

* `ResizeMethod.BILINEAR`: \[Bilinear interpolation.] (https://en.wikipedia.org/wiki/Bilinear\_interpolation)
* `ResizeMethod.NEAREST_NEIGHBOR`: \[Nearest neighbor interpolation.] (https://en.wikipedia.org/wiki/Nearest-neighbor\_interpolation)
* `ResizeMethod.BICUBIC`: \[Bicubic interpolation.] (https://en.wikipedia.org/wiki/Bicubic\_interpolation)
* `ResizeMethod.AREA`: Area interpolation.

**Args:**

* `images`: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
* `new_height`: integer.
* `new_width`: integer.
* `method`: ResizeMethod. Defaults to `ResizeMethod.BILINEAR`.
* `align_corners`: bool. If true, exactly align all 4 corners of the input and output. Defaults to `false`.

**Raises:**

* `ValueError`: if the shape of `images` is incompatible with the shape arguments to this function
* `ValueError`: if an unsupported resize method is specified.

**Returns:**

If `images` was 4-D, a 4-D float Tensor of shape `[batch, new_height, new_width, channels]`. If `images` was 3-D, a 3-D float Tensor of shape `[new_height, new_width, channels]`.

***

### `tf.image.resize_area(images, size, align_corners=None, name=None)` <a href="#resize_area" id="resize_area"></a>

Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

**Args:**

* `images`: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`. 4-D with shape `[batch, height, width, channels]`.
* `size`: A 1-D int32 Tensor of 2 elements: `new_height, new_width`. The new size for the images.
* `align_corners`: An optional `bool`. Defaults to `False`. If true, rescale input by (new\_height - 1) / (height - 1), which exactly aligns the 4 corners of images and resized images. If false, rescale by new\_height / height. Treat similarly the width dimension.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`. 4-D with shape `[batch, new_height, new_width, channels]`.

***

### `tf.image.resize_bicubic(images, size, align_corners=None, name=None)` <a href="#resize_bicubic" id="resize_bicubic"></a>

Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

**Args:**

* `images`: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`. 4-D with shape `[batch, height, width, channels]`.
* `size`: A 1-D int32 Tensor of 2 elements: `new_height, new_width`. The new size for the images.
* `align_corners`: An optional `bool`. Defaults to `False`. If true, rescale input by (new\_height - 1) / (height - 1), which exactly aligns the 4 corners of images and resized images. If false, rescale by new\_height / height. Treat similarly the width dimension.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`. 4-D with shape `[batch, new_height, new_width, channels]`.

***

### `tf.image.resize_bilinear(images, size, align_corners=None, name=None)` <a href="#resize_bilinear" id="resize_bilinear"></a>

Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

**Args:**

* `images`: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`. 4-D with shape `[batch, height, width, channels]`.
* `size`: A 1-D int32 Tensor of 2 elements: `new_height, new_width`. The new size for the images.
* `align_corners`: An optional `bool`. Defaults to `False`. If true, rescale input by (new\_height - 1) / (height - 1), which exactly aligns the 4 corners of images and resized images. If false, rescale by new\_height / height. Treat similarly the width dimension.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`. 4-D with shape `[batch, new_height, new_width, channels]`.

***

### `tf.image.resize_nearest_neighbor(images, size, align_corners=None, name=None)` <a href="#resize_nearest_neighbor" id="resize_nearest_neighbor"></a>

Resize `images` to `size` using nearest neighbor interpolation.

**Args:**

* `images`: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`. 4-D with shape `[batch, height, width, channels]`.
* `size`: A 1-D int32 Tensor of 2 elements: `new_height, new_width`. The new size for the images.
* `align_corners`: An optional `bool`. Defaults to `False`. If true, rescale input by (new\_height - 1) / (height - 1), which exactly aligns the 4 corners of images and resized images. If false, rescale by new\_height / height. Treat similarly the width dimension.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `images`. 4-D with shape `[batch, new_height, new_width, channels]`.

## Cropping

***

### `tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)` <a href="#resize_image_with_crop_or_pad" id="resize_image_with_crop_or_pad"></a>

Crops and/or pads an image to a target width and height.

Resizes an image to a target width and height by either centrally cropping the image or padding it evenly with zeros.

If `width` or `height` is greater than the specified `target_width` or `target_height` respectively, this op centrally crops along that dimension. If `width` or `height` is smaller than the specified `target_width` or `target_height` respectively, this op centrally pads with 0 along that dimension.

**Args:**

* `image`: 3-D tensor of shape \[height, width, channels]
* `target_height`: Target height.
* `target_width`: Target width.

**Raises:**

* `ValueError`: if `target_height` or `target_width` are zero or negative.

**Returns:**

Cropped and/or padded image of shape `[target_height, target_width, channels]`

***

### `tf.image.central_crop(image, central_fraction)` <a href="#central_crop" id="central_crop"></a>

Crop the central region of the image.

Remove the outer parts of an image but retain the central region of the image along each dimension. If we specify central\_fraction = 0.5, this function returns the region marked with "X" in the below diagram.

```
 --------
|        |
|  XXXX  |
|  XXXX  |
|        |   where "X" is the central 50% of the image.
 --------
```

**Args:**

* `image`: 3-D float Tensor of shape \[height, width, depth]
* `central_fraction`: float (0, 1], fraction of size to crop

**Raises:**

* `ValueError`: if central\_crop\_fraction is not within (0, 1].

**Returns:**

3-D float Tensor

***

### `tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)` <a href="#pad_to_bounding_box" id="pad_to_bounding_box"></a>

Pad `image` with zeros to the specified `height` and `width`.

Adds `offset_height` rows of zeros on top, `offset_width` columns of zeros on the left, and then pads the image on the bottom and right with zeros until it has dimensions `target_height`, `target_width`.

This op does nothing if `offset_*` is zero and the image already has size `target_height` by `target_width`.

**Args:**

* `image`: 3-D tensor with shape `[height, width, channels]`
* `offset_height`: Number of rows of zeros to add on top.
* `offset_width`: Number of columns of zeros to add on the left.
* `target_height`: Height of output image.
* `target_width`: Width of output image.

**Returns:**

3-D tensor of shape `[target_height, target_width, channels]`

**Raises:**

* `ValueError`: If the shape of `image` is incompatible with the `offset_*` or `target_*` arguments

***

### `tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)` <a href="#crop_to_bounding_box" id="crop_to_bounding_box"></a>

Crops an image to a specified bounding box.

This op cuts a rectangular part out of `image`. The top-left corner of the returned image is at `offset_height, offset_width` in `image`, and its lower-right corner is at `offset_height + target_height, offset_width + target_width`.

**Args:**

* `image`: 3-D tensor with shape `[height, width, channels]`
* `offset_height`: Vertical coordinate of the top-left corner of the result in the input.
* `offset_width`: Horizontal coordinate of the top-left corner of the result in the input.
* `target_height`: Height of the result.
* `target_width`: Width of the result.

**Returns:**

3-D tensor of image with shape `[target_height, target_width, channels]`

**Raises:**

* `ValueError`: If the shape of `image` is incompatible with the `offset_*` or `target_*` arguments

***

### `tf.image.extract_glimpse(input, size, offsets, centered=None, normalized=None, uniform_noise=None, name=None)` <a href="#extract_glimpse" id="extract_glimpse"></a>

Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location `offsets` from the input tensor. If the windows only partially overlaps the inputs, the non overlapping areas will be filled with random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height, glimpse_width, channels]`. The channels and batch dimensions are the same as that of the input tensor. The height and width of the output windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are

**Args:**

* `input`: A `Tensor` of type `float32`.
* `size`: A `Tensor` of type `int32`.
* `offsets`: A `Tensor` of type `float32`.
* `centered`: An optional `bool`. Defaults to `True`.
* `normalized`: An optional `bool`. Defaults to `True`.
* `uniform_noise`: An optional `bool`. Defaults to `True`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`.

## Flipping and Transposing

***

### `tf.image.flip_up_down(image)` <a href="#flip_up_down" id="flip_up_down"></a>

Flip an image horizontally (upside down).

Outputs the contents of `image` flipped along the first dimension, which is `height`.

See also `reverse()`.

**Args:**

* `image`: A 3-D tensor of shape `[height, width, channels].`

**Returns:**

A 3-D tensor of the same type and shape as `image`.

**Raises:**

* `ValueError`: if the shape of `image` not supported.

***

### `tf.image.random_flip_up_down(image, seed=None)` <a href="#random_flip_up_down" id="random_flip_up_down"></a>

Randomly flips an image vertically (upside down).

With a 1 in 2 chance, outputs the contents of `image` flipped along the first dimension, which is `height`. Otherwise output the image as-is.

**Args:**

* `image`: A 3-D tensor of shape `[height, width, channels].`
* `seed`: A Python integer. Used to create a random seed. See [`set_random_seed`](constant\_op.md#set\_random\_seed) for behavior.

**Returns:**

A 3-D tensor of the same type and shape as `image`.

**Raises:**

* `ValueError`: if the shape of `image` not supported.

***

### `tf.image.flip_left_right(image)` <a href="#flip_left_right" id="flip_left_right"></a>

Flip an image horizontally (left to right).

Outputs the contents of `image` flipped along the second dimension, which is `width`.

See also `reverse()`.

**Args:**

* `image`: A 3-D tensor of shape `[height, width, channels].`

**Returns:**

A 3-D tensor of the same type and shape as `image`.

**Raises:**

* `ValueError`: if the shape of `image` not supported.

***

### `tf.image.random_flip_left_right(image, seed=None)` <a href="#random_flip_left_right" id="random_flip_left_right"></a>

Randomly flip an image horizontally (left to right).

With a 1 in 2 chance, outputs the contents of `image` flipped along the second dimension, which is `width`. Otherwise output the image as-is.

**Args:**

* `image`: A 3-D tensor of shape `[height, width, channels].`
* `seed`: A Python integer. Used to create a random seed. See [`set_random_seed`](constant\_op.md#set\_random\_seed) for behavior.

**Returns:**

A 3-D tensor of the same type and shape as `image`.

**Raises:**

* `ValueError`: if the shape of `image` not supported.

***

### `tf.image.transpose_image(image)` <a href="#transpose_image" id="transpose_image"></a>

Transpose an image by swapping the first and second dimension.

See also `transpose()`.

**Args:**

* `image`: 3-D tensor of shape `[height, width, channels]`

**Returns:**

A 3-D tensor of shape `[width, height, channels]`

**Raises:**

* `ValueError`: if the shape of `image` not supported.

## Converting Between Colorspaces.

Image ops work either on individual images or on batches of images, depending on the shape of their input Tensor.

If 3-D, the shape is `[height, width, channels]`, and the Tensor represents one image. If 4-D, the shape is `[batch_size, height, width, channels]`, and the Tensor represents `batch_size` images.

Currently, `channels` can usefully be 1, 2, 3, or 4. Single-channel images are grayscale, images with 3 channels are encoded as either RGB or HSV. Images with 2 or 4 channels include an alpha channel, which has to be stripped from the image before passing the image to most image processing functions (and can be re-attached later).

Internally, images are either stored in as one `float32` per channel per pixel (implicitly, values are assumed to lie in `[0,1)`) or one `uint8` per channel per pixel (values are assumed to lie in `[0,255]`).

Tensorflow can convert between images in RGB or HSV. The conversion functions work only on float images, so you need to convert images in other formats using [`convert_image_dtype`](image.md#convert-image-dtype).

Example:

```python
# Decode an image and convert it to HSV.
rgb_image = tf.image.decode_png(...,  channels=3)
rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.image.rgb_to_hsv(rgb_image)
```

***

### `tf.image.rgb_to_grayscale(images, name=None)` <a href="#rgb_to_grayscale" id="rgb_to_grayscale"></a>

Converts one or more images from RGB to Grayscale.

Outputs a tensor of the same `DType` and rank as `images`. The size of the last dimension of the output is 1, containing the Grayscale value of the pixels.

**Args:**

* `images`: The RGB tensor to convert. Last dimension must have size 3 and should contain RGB values.
* `name`: A name for the operation (optional).

**Returns:**

The converted grayscale image(s).

***

### `tf.image.grayscale_to_rgb(images, name=None)` <a href="#grayscale_to_rgb" id="grayscale_to_rgb"></a>

Converts one or more images from Grayscale to RGB.

Outputs a tensor of the same `DType` and rank as `images`. The size of the last dimension of the output is 3, containing the RGB value of the pixels.

**Args:**

* `images`: The Grayscale tensor to convert. Last dimension must be size 1.
* `name`: A name for the operation (optional).

**Returns:**

The converted grayscale image(s).

***

### `tf.image.hsv_to_rgb(images, name=None)` <a href="#hsv_to_rgb" id="hsv_to_rgb"></a>

Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB value of the pixels. The output is only well defined if the value in `images` are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.

**Args:**

* `images`: A `Tensor` of type `float32`. 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`. `images` converted to RGB.

***

### `tf.image.rgb_to_hsv(images, name=None)` <a href="#rgb_to_hsv" id="rgb_to_hsv"></a>

Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV value of the pixels. The output is only well defined if the value in `images` are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0 corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

**Args:**

* `images`: A `Tensor` of type `float32`. 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `float32`. `images` converted to HSV.

***

### `tf.image.convert_image_dtype(image, dtype, saturate=False, name=None)` <a href="#convert_image_dtype" id="convert_image_dtype"></a>

Convert `image` to `dtype`, scaling its values if needed.

Images that are represented using floating point values are expected to have values in the range \[0,1). Image data stored in integer data types are expected to have values in the range `[0,MAX]`, where `MAX` is the largest positive representable number for the data type.

This op converts between data types, scaling the values appropriately before casting.

Note that converting from floating point inputs to integer types may lead to over/underflow problems. Set saturate to `True` to avoid such problem in problematic conversions. If enabled, saturation will clip the output into the allowed range before performing a potentially dangerous cast (and only before performing such a cast, i.e., when casting from a floating point to an integer type, and when casting from a signed to an unsigned type; `saturate` has no effect on casts between floats, or on casts that increase the type's range).

**Args:**

* `image`: An image.
* `dtype`: A `DType` to convert `image` to.
* `saturate`: If `True`, clip the input before casting (if necessary).
* `name`: A name for this operation (optional).

**Returns:**

`image`, converted to `dtype`.

## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness, contrast, hue, and saturation. Each adjustment can be done with predefined parameters or with random parameters picked from predefined intervals. Random adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of redundant conversions by first converting the images to the most natural data type and representation (RGB or HSV).

***

### `tf.image.adjust_brightness(image, delta)` <a href="#adjust_brightness" id="adjust_brightness"></a>

Adjust the brightness of RGB or Grayscale images.

This is a convenience method that converts an RGB image to float representation, adjusts its brightness, and then converts it back to the original data type. If several adjustments are chained it is advisable to minimize the number of redundant conversions.

The value `delta` is added to all components of the tensor `image`. Both `image` and `delta` are converted to `float` before adding (and `image` is scaled appropriately if it is in fixed-point representation). For regular images, `delta` should be in the range `[0,1)`, as it is added to the image in floating point representation, where pixel values are in the `[0,1)` range.

**Args:**

* `image`: A tensor.
* `delta`: A scalar. Amount to add to the pixel values.

**Returns:**

A brightness-adjusted tensor of the same shape and type as `image`.

***

### `tf.image.random_brightness(image, max_delta, seed=None)` <a href="#random_brightness" id="random_brightness"></a>

Adjust the brightness of images by a random factor.

Equivalent to `adjust_brightness()` using a `delta` randomly picked in the interval `[-max_delta, max_delta)`.

**Args:**

* `image`: An image.
* `max_delta`: float, must be non-negative.
* `seed`: A Python integer. Used to create a random seed. See [`set_random_seed`](constant\_op.md#set\_random\_seed) for behavior.

**Returns:**

The brightness-adjusted image.

**Raises:**

* `ValueError`: if `max_delta` is negative.

***

### `tf.image.adjust_contrast(images, contrast_factor)` <a href="#adjust_contrast" id="adjust_contrast"></a>

Adjust contrast of RGB or grayscale images.

This is a convenience method that converts an RGB image to float representation, adjusts its contrast, and then converts it back to the original data type. If several adjustments are chained it is advisable to minimize the number of redundant conversions.

`images` is a tensor of at least 3 dimensions. The last 3 dimensions are interpreted as `[height, width, channels]`. The other dimensions only represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, this Op computes the mean of the image pixels in the channel and then adjusts each component `x` of each pixel to `(x - mean) * contrast_factor + mean`.

**Args:**

* `images`: Images to adjust. At least 3-D.
* `contrast_factor`: A float multiplier for adjusting contrast.

**Returns:**

The contrast-adjusted image or images.

***

### `tf.image.random_contrast(image, lower, upper, seed=None)` <a href="#random_contrast" id="random_contrast"></a>

Adjust the contrast of an image by a random factor.

Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly picked in the interval `[lower, upper]`.

**Args:**

* `image`: An image tensor with 3 or more dimensions.
* `lower`: float. Lower bound for the random contrast factor.
* `upper`: float. Upper bound for the random contrast factor.
* `seed`: A Python integer. Used to create a random seed. See [`set_random_seed`](constant\_op.md#set\_random\_seed) for behavior.

**Returns:**

The contrast-adjusted tensor.

**Raises:**

* `ValueError`: if `upper <= lower` or if `lower < 0`.

***

### `tf.image.adjust_hue(image, delta, name=None)` <a href="#adjust_hue" id="adjust_hue"></a>

Adjust hue of an RGB image.

This is a convenience method that converts an RGB image to float representation, converts it to HSV, add an offset to the hue channel, converts back to RGB and then back to the original data type. If several adjustments are chained it is advisable to minimize the number of redundant conversions.

`image` is an RGB image. The image hue is adjusted by converting the image to HSV and rotating the hue channel (H) by `delta`. The image is then converted back to RGB.

`delta` must be in the interval `[-1, 1]`.

**Args:**

* `image`: RGB image or images. Size of the last dimension must be 3.
* `delta`: float. How much to add to the hue channel.
* `name`: A name for this operation (optional).

**Returns:**

Adjusted image(s), same shape and DType as `image`.

***

### `tf.image.random_hue(image, max_delta, seed=None)` <a href="#random_hue" id="random_hue"></a>

Adjust the hue of an RGB image by a random factor.

Equivalent to `adjust_hue()` but uses a `delta` randomly picked in the interval `[-max_delta, max_delta]`.

`max_delta` must be in the interval `[0, 0.5]`.

**Args:**

* `image`: RGB image or images. Size of the last dimension must be 3.
* `max_delta`: float. Maximum value for the random delta.
* `seed`: An operation-specific seed. It will be used in conjunction with the graph-level seed to determine the real seeds that will be used in this operation. Please see the documentation of set\_random\_seed for its interaction with the graph-level random seed.

**Returns:**

3-D float tensor of shape `[height, width, channels]`.

**Raises:**

* `ValueError`: if `max_delta` is invalid.

***

### `tf.image.adjust_saturation(image, saturation_factor, name=None)` <a href="#adjust_saturation" id="adjust_saturation"></a>

Adjust saturation of an RGB image.

This is a convenience method that converts an RGB image to float representation, converts it to HSV, add an offset to the saturation channel, converts back to RGB and then back to the original data type. If several adjustments are chained it is advisable to minimize the number of redundant conversions.

`image` is an RGB image. The image saturation is adjusted by converting the image to HSV and multiplying the saturation (S) channel by `saturation_factor` and clipping. The image is then converted back to RGB.

**Args:**

* `image`: RGB image or images. Size of the last dimension must be 3.
* `saturation_factor`: float. Factor to multiply the saturation by.
* `name`: A name for this operation (optional).

**Returns:**

Adjusted image(s), same shape and DType as `image`.

***

### `tf.image.random_saturation(image, lower, upper, seed=None)` <a href="#random_saturation" id="random_saturation"></a>

Adjust the saturation of an RGB image by a random factor.

Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly picked in the interval `[lower, upper]`.

**Args:**

* `image`: RGB image or images. Size of the last dimension must be 3.
* `lower`: float. Lower bound for the random saturation factor.
* `upper`: float. Upper bound for the random saturation factor.
* `seed`: An operation-specific seed. It will be used in conjunction with the graph-level seed to determine the real seeds that will be used in this operation. Please see the documentation of set\_random\_seed for its interaction with the graph-level random seed.

**Returns:**

Adjusted image(s), same shape and DType as `image`.

**Raises:**

* `ValueError`: if `upper <= lower` or if `lower < 0`.

***

### `tf.image.per_image_whitening(image)` <a href="#per_image_whitening" id="per_image_whitening"></a>

Linearly scales `image` to have zero mean and unit norm.

This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average of all values in image, and `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

`stddev` is the standard deviation of all values in `image`. It is capped away from zero to protect against division by 0 when handling uniform images.

Note that this implementation is limited:

* It only whitens based on the statistics of an individual image.
* It does not take into account the covariance structure.

**Args:**

* `image`: 3-D tensor of shape `[height, width, channels]`.

**Returns:**

The whitened image with same shape as `image`.

**Raises:**

* `ValueError`: if the shape of 'image' is incompatible with this function.

## Working with Bounding Boxes

***

### `tf.image.draw_bounding_boxes(images, boxes, name=None)` <a href="#draw_bounding_boxes" id="draw_bounding_boxes"></a>

Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding boxes specified by the locations in `boxes`. The coordinates of the each bounding box in `boxes are encoded as` \[y\_min, x\_min, y\_max, x\_max]`. The bounding box coordinates are floats in` \[0.0, 1.0]\` relative to the width and height of the underlying image.

For example, if an image is 100 x 200 pixels and the bounding box is `[0.1, 0.5, 0.2, 0.9]`, the bottom-left and upper-right coordinates of the bounding box will be `(10, 40)` to `(50, 180)`.

Parts of the bounding box may fall outside the image.

**Args:**

* `images`: A `Tensor`. Must be one of the following types: `float32`, `half`. 4-D with shape `[batch, height, width, depth]`. A batch of images.
* `boxes`: A `Tensor` of type `float32`. 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `images`. 4-D with the same shape as `images`. The batch of input images with bounding boxes drawn on the images.

***

### `tf.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed=None, seed2=None, min_object_covered=None, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None)` <a href="#sample_distorted_bounding_box" id="sample_distorted_bounding_box"></a>

Generate a single randomly distorted bounding box for an image.

Bounding box annotations are often supplied in addition to ground-truth labels in image recognition or object localization tasks. A common technique for training such a system is to randomly distort an image while preserving its content, i.e. _data augmentation_. This Op outputs a randomly distorted localization of an object, i.e. bounding box, given an `image_size`, `bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the original image. The output is returned as 3 tensors: `begin`, `size` and `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the image. The latter may be supplied to `tf.image.draw_bounding_box` to visualize what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and height of the underlying image.

For example,

```
# Generate a single distorted bounding box.
begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
    tf.shape(image),
    bounding_boxes=bounding_boxes)

# Draw the bounding box in an image summary.
image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                              bbox_for_draw)
tf.image_summary('images_with_box', image_with_box)

# Employ the bounding box to distort the image.
distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting `use_image_if_no_bounding_boxes = true` will assume there is a single implicit bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is false and no bounding boxes are supplied, an error is raised.

**Args:**

* `image_size`: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
* `bounding_boxes`: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]` describing the N bounding boxes associated with the image.
* `seed`: An optional `int`. Defaults to `0`. If either `seed` or `seed2` are set to non-zero, the random number generator is seeded by the given `seed`. Otherwise, it is seeded by a random seed.
* `seed2`: An optional `int`. Defaults to `0`. A second seed to avoid seed collision.
* `min_object_covered`: An optional `float`. Defaults to `0.1`. The cropped area of the image must contain at least this fraction of any bounding box supplied.
* `aspect_ratio_range`: An optional list of `floats`. Defaults to `[0.75, 1.33]`. The cropped area of the image must have an aspect ratio = width / height within this range.
* `area_range`: An optional list of `floats`. Defaults to `[0.05, 1]`. The cropped area of the image must contain a fraction of the supplied image within in this range.
* `max_attempts`: An optional `int`. Defaults to `100`. Number of attempts at generating a cropped region of the image of the specified constraints. After `max_attempts` failures, return the entire image.
* `use_image_if_no_bounding_boxes`: An optional `bool`. Defaults to `False`. Controls behavior if no bounding boxes supplied. If true, assume an implicit bounding box covering the whole input. If false, raise an error.
* `name`: A name for the operation (optional).

**Returns:**

A tuple of `Tensor` objects (begin, size, bboxes).

* `begin`: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to `tf.slice`.
* `size`: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[target_height, target_width, -1]`. Provide as input to `tf.slice`.
* `bboxes`: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing the distorted bounding box. Provide as input to `tf.image.draw_bounding_boxes`.
