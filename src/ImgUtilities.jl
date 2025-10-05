using CUDA
using Images

function flatten_image(img::AbstractArray{T}) where T<:Real
    if ndims(img) == 3
        channels, w, h = size(img)
    elseif ndims(img) == 2
        w, h = size(img)
        channels = 1
    else
        throw("Unsupported image shape $(size(img))")
    end
    original_shape = size(img)
    img_rearranged = reshape(img, (channels, w * h))

    return img_rearranged, original_shape
end

@inline function rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
    return (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
end

function restore_image(img::AbstractArray{T}, original_shape::Tuple) where T<:Real
    channels, w, h = original_shape
    return reshape(img, (channels, w, h))
end

function load_rgb(filepath::String; cuda::Bool=false, size::Tuple=())
    img = Float64.(channelview(imresize(load(filepath), size, ratio=1.0)))


    img, original_dims = flatten_image(img)
    if cuda
        return CuArray(img), original_dims
    end
    return img, original_dims
end
function apply_colormap(img::AbstractMatrix{T}, colormap::AbstractVector{Int}) where T<:Real
    imgcopy = img[:, colormap]
end
function save_image(path::String, flattened_image::AbstractMatrix{T}, original_shape::Tuple) where T<:Real
    img = colorview(RGB, restore_image(flattened_image, original_shape))
    save(path, img)
end

