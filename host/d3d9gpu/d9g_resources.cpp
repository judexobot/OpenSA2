// d9g_resources.cpp: textures, surfaces, vertex and index buffers, declarations and
// shaders, with D3D9's lifetime rules (DXVK's D3D9DeviceChild: a public and a private
// count; a texture level's surface counts on its texture) and DXVK 3.1.1's lock rules
// (d3d9_surface.cpp LockRect, d3d9_device.cpp LockImage/UnlockImage, LockBuffer/
// UnlockBuffer).
#include "d9g.h"

#include <algorithm>
#include <cstring>

namespace d9g {

// ------------------------------------------------------------------ Child
template <class Stub, class Com, int I>
Child<Stub, Com, I>::Child(Device* d) : dev_(d) {}

template <class Stub, class Com, int I>
Child<Stub, Com, I>::~Child() {
    if (dev_) dev_->delist(this);
}

template <class Stub, class Com, int I>
HRESULT STDMETHODCALLTYPE Child<Stub, Com, I>::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return answer(I, 0, E_POINTER);
    *ppv = nullptr;
    const GUID unk = __uuidof(IUnknown), own = __uuidof(Com);
    if (!std::memcmp(&riid, &unk, sizeof(GUID)) || !std::memcmp(&riid, &own, sizeof(GUID))) {
        this->AddRef();
        *ppv = static_cast<Com*>(this);
        return answer(I, 0, S_OK);
    }
    return answer(I, 0, E_NOINTERFACE);
}

template <class Stub, class Com, int I>
ULONG STDMETHODCALLTYPE Child<Stub, Com, I>::AddRef() {
    return ++pub_;
}

template <class Stub, class Com, int I>
ULONG STDMETHODCALLTYPE Child<Stub, Com, I>::Release() {
    if (!pub_) return 0;
    const ULONG n = --pub_;
    if (!n && !priv_ && !owned_) delete this;
    return n;
}

template <class Stub, class Com, int I>
void Child<Stub, Com, I>::add_priv() {
    ++priv_;
}

template <class Stub, class Com, int I>
void Child<Stub, Com, I>::release_priv() {
    if (priv_ && --priv_ == 0 && !pub_ && !owned_) delete this;
}

template class Child<D9gStub_IDirect3DSurface9, IDirect3DSurface9, I_SURFACE>;
template class Child<D9gStub_IDirect3DTexture9, IDirect3DTexture9, I_TEXTURE>;
template class Child<D9gStub_IDirect3DVertexBuffer9, IDirect3DVertexBuffer9, I_VB>;
template class Child<D9gStub_IDirect3DIndexBuffer9, IDirect3DIndexBuffer9, I_IB>;
template class Child<D9gStub_IDirect3DVertexDeclaration9, IDirect3DVertexDeclaration9, I_DECL>;
template class Child<D9gStub_IDirect3DVertexShader9, IDirect3DVertexShader9, I_VS>;
template class Child<D9gStub_IDirect3DPixelShader9, IDirect3DPixelShader9, I_PS>;
template class Child<D9gStub_IDirect3DCubeTexture9, IDirect3DCubeTexture9, I_CUBE>;
template class Child<D9gStub_IDirect3DVolumeTexture9, IDirect3DVolumeTexture9, I_VOLUME_TEX>;

namespace {

// LockImage (d3d9_device.cpp:4950-5207) for one subresource `sub` (a level of `level`), after
// Surface::LockRect's rectangle checks (d3d9_surface.cpp:128-199). A 2D image's subresource is its level.
HRESULT lock_image(Image& img, uint32_t sub, uint32_t level, D3DLOCKED_RECT* lr, const RECT* rect, DWORD flags) {
    if (!lr) return D3DERR_INVALIDCALL;
    if (img.pool == D3DPOOL_DEFAULT) { lr->pBits = nullptr; lr->Pitch = 0; }
    const FormatInfo& fi = *img.fi;
    const uint32_t lw = level_dim(img.w, level), lh = level_dim(img.h, level);
    if (rect) {
        const bool block = fi.block_bytes != 0;
        const bool sysmem_surface = img.pool == D3DPOOL_SYSTEMMEM && !img.is_texture;
        if ((block && img.pool == D3DPOOL_DEFAULT) || sysmem_surface) {
            const LONG width = rect->right - rect->left, height = rect->bottom - rect->top;
            const bool bad_dims = sysmem_surface ? (width < 0 || height < 0) : (width <= 0 || height <= 0);
            if (rect->left < 0 || rect->right < 0 || rect->top < 0 || rect->bottom < 0 || bad_dims ||
                UINT(rect->right) > lw || UINT(rect->bottom) > lh)
                return D3DERR_INVALIDCALL;
        }
    }
    if (img.locked[sub]) return D3DERR_INVALIDCALL;                       // :4962
    if ((flags & (D3DLOCK_DISCARD | D3DLOCK_READONLY)) == (D3DLOCK_DISCARD | D3DLOCK_READONLY) &&
        img.pool == D3DPOOL_DEFAULT)
        return D3DERR_INVALIDCALL;                                          // :4967
    if (!img.lockable) return D3DERR_INVALIDCALL;                          // :4982
    if (rect && level == 0 && fi.block_bytes && img.pool == D3DPOOL_DEFAULT) {   // :4985-5003
        const auto bad = [](LONG v) { return v && (v & 3); };
        if (bad(rect->left) || bad(rect->top) || bad(rect->right) || bad(rect->bottom)) return D3DERR_INVALIDCALL;
    }
    if (img.store.size() <= sub || img.store[sub].empty()) {
        not_implemented_value("LockRect of a level with no CPU store (" + std::string(fi.name) + ")");
        return D3DERR_INVALIDCALL;
    }
    const uint32_t pitch = level_pitch(lw, fi);
    size_t offset = 0;
    if (rect) {
        const uint32_t bw = fi.block_bytes ? 4 : 1;
        const uint32_t elem = fi.block_bytes ? fi.block_bytes : fi.pixel_bytes;
        offset = size_t(uint32_t(rect->top) / bw) * pitch + size_t(uint32_t(rect->left) / bw) * elem;
        if (offset >= img.store[sub].size()) {
            not_implemented_value("LockRect rectangle starting past the level's store");
            return D3DERR_INVALIDCALL;
        }
    }
    img.locked[sub] = 1;
    img.lock_flags[sub] = flags;
    lr->pBits = img.store[sub].data() + offset;
    lr->Pitch = INT(pitch);
    return D3D_OK;
}

// UnlockImage (d3d9_device.cpp:5212-5256): an unlocked level of a 2D texture answers
// D3D_OK (D3DRTYPE_TEXTURE alone, :5219-5224), of a surface, a cube or a volume
// D3DERR_INVALIDCALL. The level goes to the GPU at once (DXVK uploads MANAGED textures at
// draw time; the bytes are the same).
HRESULT unlock_image(Device* dev, Image& img, uint32_t sub) {
    if (!img.locked[sub]) return img.is_texture && img.type == TexType::flat ? D3D_OK : D3DERR_INVALIDCALL;
    img.locked[sub] = 0;
    if (!(img.lock_flags[sub] & D3DLOCK_READONLY) && img.gpu && dev) dev->upload_image_subresource(img, sub);
    return D3D_OK;
}

// D3D9Volume::LockBox's box checks (d3d9_volume.cpp:73-96), then LockImage
// (d3d9_device.cpp:4950-5207) for a volume's level: the block-alignment rule at level 0 of a
// volume whatever its pool (:4995-5010), RowPitch the level's packed pitch, SlicePitch that
// times its block rows (:5151-5152), and the box's front slice, top row and left element
// (CalcImageLockOffset :4919-4947).
HRESULT lock_box(Image& img, uint32_t level, D3DLOCKED_BOX* lb, const D3DBOX* box, DWORD flags) {
    if (!lb) return D3DERR_INVALIDCALL;
    lb->pBits = nullptr;
    lb->RowPitch = 0;
    lb->SlicePitch = 0;
    const FormatInfo& fi = *img.fi;
    const uint32_t lw = level_dim(img.w, level), lh = level_dim(img.h, level), ld = img.level_depth(level);
    if (box) {
        if (LONG(box->Right) - LONG(box->Left) <= 0 || LONG(box->Bottom) - LONG(box->Top) <= 0 ||
            LONG(box->Back) - LONG(box->Front) <= 0 || box->Right > lw || box->Bottom > lh || box->Back > ld)
            return D3DERR_INVALIDCALL;
    }
    if (img.locked[level]) return D3DERR_INVALIDCALL;
    if ((flags & (D3DLOCK_DISCARD | D3DLOCK_READONLY)) == (D3DLOCK_DISCARD | D3DLOCK_READONLY) &&
        img.pool == D3DPOOL_DEFAULT)
        return D3DERR_INVALIDCALL;
    if (!img.lockable) return D3DERR_INVALIDCALL;
    if (box && level == 0 && fi.block_bytes) {
        const auto bad = [](UINT v) { return v && (v & 3); };
        if (bad(box->Left) || bad(box->Top) || bad(box->Right) || bad(box->Bottom)) return D3DERR_INVALIDCALL;
    }
    if (img.store.size() <= level || img.store[level].empty()) {
        not_implemented_value("LockBox of a level with no CPU store (" + std::string(fi.name) + ")");
        return D3DERR_INVALIDCALL;
    }
    const uint32_t pitch = level_pitch(lw, fi), slice = pitch * level_rows(lh, fi);
    size_t offset = 0;
    if (box) {
        const uint32_t bw = fi.block_bytes ? 4 : 1;
        const uint32_t elem = fi.block_bytes ? fi.block_bytes : fi.pixel_bytes;
        offset = size_t(box->Front) * slice + size_t(box->Top / bw) * pitch + size_t(box->Left / bw) * elem;
    }
    img.locked[level] = 1;
    img.lock_flags[level] = flags;
    lb->pBits = img.store[level].data() + offset;
    lb->RowPitch = INT(pitch);
    lb->SlicePitch = INT(slice);
    return D3D_OK;
}

void fill_desc(const Image& img, uint32_t level, D3DSURFACE_DESC* d) {
    d->Format = D3DFORMAT(img.d3dfmt);
    d->Type = D3DRTYPE_SURFACE;
    d->Usage = img.usage;
    d->Pool = D3DPOOL(img.pool);
    d->MultiSampleType = D3DMULTISAMPLE_NONE;
    d->MultiSampleQuality = 0;
    d->Width = level_dim(img.w, level);
    d->Height = level_dim(img.h, level);
}

}  // namespace

// ------------------------------------------------------------------ Surface
Surface::Surface(Device* d, Texture* container, Image* img, uint32_t level)
    : Child(d), container_(container), img_(img), level_(level) {}

Surface::Surface(Device* d, CubeTexture* cube, Image* img, uint32_t face, uint32_t level)
    : Child(d), container_(nullptr), cube_(cube), img_(img), level_(level), face_(face) {}

Surface::Surface(Device* d, std::unique_ptr<Image> own, bool enlist)
    : Child(d), container_(nullptr), img_(own.get()), level_(0), own_(std::move(own)) {
    if (dev_ && enlist) dev_->enlist(this, "surface");
}

Surface::~Surface() {
    if (own_ && own_->gpu && dev_) dev_->release_gpu_texture(own_->gpu);
    if (container_ || cube_) dev_ = nullptr;   // a level or a face is never enlisted: its texture is
}

ULONG STDMETHODCALLTYPE Surface::AddRef() {
    return container_ ? container_->AddRef() : cube_ ? cube_->AddRef() : Child::AddRef();
}
ULONG STDMETHODCALLTYPE Surface::Release() {
    return container_ ? container_->Release() : cube_ ? cube_->Release() : Child::Release();
}
void Surface::add_priv() { container_ ? container_->add_priv() : cube_ ? cube_->add_priv() : Child::add_priv(); }
void Surface::release_priv() {
    container_ ? container_->release_priv() : cube_ ? cube_->release_priv() : Child::release_priv();
}

void Surface::orphan() {
    if (own_ && own_->gpu && dev_) { dev_->release_gpu_texture(own_->gpu); own_->gpu = 0; }
    dev_ = nullptr;
}

HRESULT STDMETHODCALLTYPE Surface::GetDevice(IDirect3DDevice9** pp) {
    if (!pp) return answer(I_SURFACE, 3, D3DERR_INVALIDCALL);
    *pp = dev_;
    if (dev_) dev_->AddRef();
    return answer(I_SURFACE, 3, dev_ ? D3D_OK : D3DERR_INVALIDCALL);
}

D3DRESOURCETYPE STDMETHODCALLTYPE Surface::GetType() {
    answer(I_SURFACE, 10, D3D_OK);
    return D3DRTYPE_SURFACE;
}

HRESULT STDMETHODCALLTYPE Surface::GetDesc(D3DSURFACE_DESC* d) {
    if (!d) return answer(I_SURFACE, 12, D3DERR_INVALIDCALL);
    fill_desc(*img_, level_, d);
    return answer(I_SURFACE, 12, D3D_OK);
}

HRESULT STDMETHODCALLTYPE Surface::LockRect(D3DLOCKED_RECT* lr, const RECT* rect, DWORD flags) {
    return answer(I_SURFACE, 13, lock_image(*img_, subresource(), level_, lr, rect, flags));
}

// A cube face's surface: its cube's device (a face is never enlisted, so dev_ is the cube's).
HRESULT STDMETHODCALLTYPE Surface::UnlockRect() {
    return answer(I_SURFACE, 14, unlock_image(cube_ ? cube_->device() : dev_, *img_, subresource()));
}

// ------------------------------------------------------------------ Texture
Texture::Texture(Device* d, std::unique_ptr<Image> img) : Child(d), img_(std::move(img)) {
    img_->is_texture = true;
    for (uint32_t l = 0; l < img_->levels; l++) levels_.push_back(new Surface(d, this, img_.get(), l));
    if (dev_) dev_->enlist(this, "texture");
}

Texture::~Texture() {
    if (img_->gpu && dev_) dev_->release_gpu_texture(img_->gpu);
    for (Surface* s : levels_) delete s;
}

void Texture::orphan() {
    if (img_->gpu && dev_) { dev_->release_gpu_texture(img_->gpu); img_->gpu = 0; }
    dev_ = nullptr;
}

HRESULT STDMETHODCALLTYPE Texture::GetDevice(IDirect3DDevice9** pp) {
    if (!pp) return answer(I_TEXTURE, 3, D3DERR_INVALIDCALL);
    *pp = dev_;
    if (dev_) dev_->AddRef();
    return answer(I_TEXTURE, 3, dev_ ? D3D_OK : D3DERR_INVALIDCALL);
}

// D3D9Texture2D's SetLOD (d3d9_texture.h:60-71): min(LODNew, MipLevels - 1), the old one returned.
DWORD STDMETHODCALLTYPE Texture::SetLOD(DWORD lod) {
    answer(I_TEXTURE, 11, D3D_OK);
    const DWORD old = lod_;
    lod_ = std::min<DWORD>(lod, img_->levels - 1);
    return old;
}

HRESULT STDMETHODCALLTYPE Texture::GetLevelDesc(UINT level, D3DSURFACE_DESC* d) {
    if (level >= img_->levels || !d) return answer(I_TEXTURE, 17, D3DERR_INVALIDCALL);
    fill_desc(*img_, level, d);
    return answer(I_TEXTURE, 17, D3D_OK);
}

// d3d9_texture.cpp:63-74.
HRESULT STDMETHODCALLTYPE Texture::GetSurfaceLevel(UINT level, IDirect3DSurface9** pp) {
    if (pp) *pp = nullptr;
    if (level >= img_->levels || !pp) return answer(I_TEXTURE, 18, D3DERR_INVALIDCALL);
    levels_[level]->AddRef();
    *pp = levels_[level];
    return answer(I_TEXTURE, 18, D3D_OK);
}

// d3d9_texture.cpp:77-90.
HRESULT STDMETHODCALLTYPE Texture::LockRect(UINT level, D3DLOCKED_RECT* lr, const RECT* rect, DWORD flags) {
    if (level >= img_->levels) return answer(I_TEXTURE, 19, D3DERR_INVALIDCALL);
    return answer(I_TEXTURE, 19, lock_image(*img_, level, level, lr, rect, flags));
}

HRESULT STDMETHODCALLTYPE Texture::UnlockRect(UINT level) {
    if (level >= img_->levels) return answer(I_TEXTURE, 20, D3DERR_INVALIDCALL);
    return answer(I_TEXTURE, 20, unlock_image(dev_, *img_, level));
}

// ------------------------------------------------------------------ CubeTexture
// D3D9TextureCube (d3d9_texture.cpp:203-328) on D3D9BaseTexture (d3d9_texture.h): the subresource
// of (face, level) is face * levels + level (CalcSubresource), each a Surface whose counts are the cube's.
CubeTexture::CubeTexture(Device* d, std::unique_ptr<Image> img) : Child(d), img_(std::move(img)) {
    img_->is_texture = true;
    for (uint32_t face = 0; face < 6; face++)
        for (uint32_t l = 0; l < img_->levels; l++) faces_.push_back(new Surface(d, this, img_.get(), face, l));
    if (dev_) dev_->enlist(this, "cube_texture");
}

CubeTexture::~CubeTexture() {
    if (img_->gpu && dev_) dev_->release_gpu_texture(img_->gpu);
    for (Surface* s : faces_) delete s;
}

void CubeTexture::orphan() {
    if (img_->gpu && dev_) { dev_->release_gpu_texture(img_->gpu); img_->gpu = 0; }
    dev_ = nullptr;
}

HRESULT STDMETHODCALLTYPE CubeTexture::GetDevice(IDirect3DDevice9** pp) {
    if (!pp) return answer(I_CUBE, 3, D3DERR_INVALIDCALL);
    *pp = dev_;
    if (dev_) dev_->AddRef();
    return answer(I_CUBE, 3, dev_ ? D3D_OK : D3DERR_INVALIDCALL);
}

D3DRESOURCETYPE STDMETHODCALLTYPE CubeTexture::GetType() {
    answer(I_CUBE, 10, D3D_OK);
    return D3DRTYPE_CUBETEXTURE;
}

// D3D9BaseTexture::SetLOD (d3d9_texture.h:60-71): min(LODNew, MipLevels - 1), the old one returned.
DWORD STDMETHODCALLTYPE CubeTexture::SetLOD(DWORD lod) {
    answer(I_CUBE, 11, D3D_OK);
    const DWORD old = lod_;
    lod_ = std::min<DWORD>(lod, img_->levels - 1);
    return old;
}

DWORD STDMETHODCALLTYPE CubeTexture::GetLOD() {
    answer(I_CUBE, 12, D3D_OK);
    return lod_;
}

// ExposedMipLevels (d3d9_common_texture.h): the levels made (no autogen here).
DWORD STDMETHODCALLTYPE CubeTexture::GetLevelCount() {
    answer(I_CUBE, 13, D3D_OK);
    return img_->levels;
}

// d3d9_texture.cpp:250-255: face 0's level (every face's is the same size).
HRESULT STDMETHODCALLTYPE CubeTexture::GetLevelDesc(UINT level, D3DSURFACE_DESC* d) {
    if (level >= img_->levels || !d) return answer(I_CUBE, 17, D3DERR_INVALIDCALL);
    fill_desc(*img_, level, d);
    return answer(I_CUBE, 17, D3D_OK);
}

// d3d9_texture.cpp:258-270.
HRESULT STDMETHODCALLTYPE CubeTexture::GetCubeMapSurface(D3DCUBEMAP_FACES face, UINT level, IDirect3DSurface9** pp) {
    if (pp) *pp = nullptr;
    if (level >= img_->levels || uint32_t(face) >= 6 || !pp) return answer(I_CUBE, 18, D3DERR_INVALIDCALL);
    Surface* s = faces_[img_->subresource(uint32_t(face), level)];
    s->AddRef();
    *pp = s;
    return answer(I_CUBE, 18, D3D_OK);
}

// d3d9_texture.cpp:273-286: the face's surface's LockRect and UnlockRect.
HRESULT STDMETHODCALLTYPE CubeTexture::LockRect(D3DCUBEMAP_FACES face, UINT level, D3DLOCKED_RECT* lr, const RECT* rect,
                                                DWORD flags) {
    if (uint32_t(face) > D3DCUBEMAP_FACE_NEGATIVE_Z || level >= img_->levels) return answer(I_CUBE, 19, D3DERR_INVALIDCALL);
    return answer(I_CUBE, 19, lock_image(*img_, img_->subresource(uint32_t(face), level), level, lr, rect, flags));
}

HRESULT STDMETHODCALLTYPE CubeTexture::UnlockRect(D3DCUBEMAP_FACES face, UINT level) {
    if (uint32_t(face) > D3DCUBEMAP_FACE_NEGATIVE_Z || level >= img_->levels) return answer(I_CUBE, 20, D3DERR_INVALIDCALL);
    return answer(I_CUBE, 20, unlock_image(dev_, *img_, img_->subresource(uint32_t(face), level)));
}

// ------------------------------------------------------------------ VolumeTexture
// D3D9Texture3D (d3d9_texture.cpp:111-200): one layer, each level's subresource its level.
VolumeTexture::VolumeTexture(Device* d, std::unique_ptr<Image> img) : Child(d), img_(std::move(img)) {
    img_->is_texture = true;
    if (dev_) dev_->enlist(this, "volume_texture");
}

VolumeTexture::~VolumeTexture() {
    if (img_->gpu && dev_) dev_->release_gpu_texture(img_->gpu);
}

void VolumeTexture::orphan() {
    if (img_->gpu && dev_) { dev_->release_gpu_texture(img_->gpu); img_->gpu = 0; }
    dev_ = nullptr;
}

HRESULT STDMETHODCALLTYPE VolumeTexture::GetDevice(IDirect3DDevice9** pp) {
    if (!pp) return answer(I_VOLUME_TEX, 3, D3DERR_INVALIDCALL);
    *pp = dev_;
    if (dev_) dev_->AddRef();
    return answer(I_VOLUME_TEX, 3, dev_ ? D3D_OK : D3DERR_INVALIDCALL);
}

D3DRESOURCETYPE STDMETHODCALLTYPE VolumeTexture::GetType() {
    answer(I_VOLUME_TEX, 10, D3D_OK);
    return D3DRTYPE_VOLUMETEXTURE;
}

DWORD STDMETHODCALLTYPE VolumeTexture::SetLOD(DWORD lod) {
    answer(I_VOLUME_TEX, 11, D3D_OK);
    const DWORD old = lod_;
    lod_ = std::min<DWORD>(lod, img_->levels - 1);
    return old;
}

DWORD STDMETHODCALLTYPE VolumeTexture::GetLOD() {
    answer(I_VOLUME_TEX, 12, D3D_OK);
    return lod_;
}

DWORD STDMETHODCALLTYPE VolumeTexture::GetLevelCount() {
    answer(I_VOLUME_TEX, 13, D3D_OK);
    return img_->levels;
}

// d3d9_texture.cpp:150-155, D3D9Volume::GetDesc (d3d9_volume.cpp:54-71).
HRESULT STDMETHODCALLTYPE VolumeTexture::GetLevelDesc(UINT level, D3DVOLUME_DESC* d) {
    if (level >= img_->levels || !d) return answer(I_VOLUME_TEX, 17, D3DERR_INVALIDCALL);
    d->Format = D3DFORMAT(img_->d3dfmt);
    d->Type = D3DRTYPE_VOLUME;
    d->Usage = img_->usage;
    d->Pool = D3DPOOL(img_->pool);
    d->Width = level_dim(img_->w, level);
    d->Height = level_dim(img_->h, level);
    d->Depth = img_->level_depth(level);
    return answer(I_VOLUME_TEX, 17, D3D_OK);
}

// d3d9_texture.cpp:172-185.
HRESULT STDMETHODCALLTYPE VolumeTexture::LockBox(UINT level, D3DLOCKED_BOX* lb, const D3DBOX* box, DWORD flags) {
    if (level >= img_->levels) return answer(I_VOLUME_TEX, 19, D3DERR_INVALIDCALL);
    return answer(I_VOLUME_TEX, 19, lock_box(*img_, level, lb, box, flags));
}

HRESULT STDMETHODCALLTYPE VolumeTexture::UnlockBox(UINT level) {
    if (level >= img_->levels) return answer(I_VOLUME_TEX, 20, D3DERR_INVALIDCALL);
    return answer(I_VOLUME_TEX, 20, unlock_image(dev_, *img_, level));
}

// ------------------------------------------------------------------ buffers
template <class Stub, class Com, int I, class Desc>
BufferBase<Stub, Com, I, Desc>::BufferBase(Device* d, uint32_t len, uint32_t usage, uint32_t pool, uint32_t fmt,
                                           uint32_t gpu)
    : Child<Stub, Com, I>(d), len_(len), usage_(usage), pool_(pool), fmt_(fmt), gpu_(gpu), store_(len, 0) {
    if (this->dev_) this->dev_->enlist(this, I == I_VB ? "vertex_buffer" : "index_buffer");
}

template <class Stub, class Com, int I, class Desc>
BufferBase<Stub, Com, I, Desc>::~BufferBase() {
    if (gpu_ && this->dev_) this->dev_->gpu()->release_buffer(gpu_);
}

template <class Stub, class Com, int I, class Desc>
void BufferBase<Stub, Com, I, Desc>::orphan() {
    if (gpu_ && this->dev_) { this->dev_->gpu()->release_buffer(gpu_); gpu_ = 0; }
    this->dev_ = nullptr;
}

template <class Stub, class Com, int I, class Desc>
HRESULT STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::GetDevice(IDirect3DDevice9** pp) {
    if (!pp) return answer(I, 3, D3DERR_INVALIDCALL);
    *pp = this->dev_;
    if (this->dev_) this->dev_->AddRef();
    return answer(I, 3, this->dev_ ? D3D_OK : D3DERR_INVALIDCALL);
}

template <class Stub, class Com, int I, class Desc>
D3DRESOURCETYPE STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::GetType() {
    answer(I, 10, D3D_OK);
    return I == I_VB ? D3DRTYPE_VERTEXBUFFER : D3DRTYPE_INDEXBUFFER;
}

// LockBuffer (d3d9_device.cpp:5451-5584): no bounds check; the pointer is the store
// plus OffsetToLock, unclamped. The range locked (all of it when SizeToLock is 0) is
// dirty unless READONLY, and goes to the GPU when the last lock is undone.
template <class Stub, class Com, int I, class Desc>
HRESULT STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::Lock(UINT offset, UINT size, void** data, DWORD flags) {
    if (!data) return answer(I, 11, D3DERR_INVALIDCALL);
    if (offset > len_) {
        // DXVK hands out a pointer past its mapping; writing through it is undefined.
        not_implemented_value("Lock offset past the buffer's end");
        *data = nullptr;
        return answer(I, 11, D3DERR_INVALIDCALL);
    }
    if ((flags & D3DLOCK_READONLY) && pool_ != D3DPOOL_MANAGED) flags &= ~DWORD(D3DLOCK_READONLY);   // :5493
    if (!(flags & D3DLOCK_READONLY)) {
        const uint32_t lo = size ? offset : 0, hi = size ? offset + std::min(size, len_ - offset) : len_;
        if (dirty_hi_ <= dirty_lo_) { dirty_lo_ = lo; dirty_hi_ = hi; }
        else { dirty_lo_ = std::min(dirty_lo_, lo); dirty_hi_ = std::max(dirty_hi_, hi); }
    }
    index_ranges_.invalidate();
    *data = store_.data() + offset;
    lock_count_++;
    return answer(I, 11, D3D_OK);
}

// UnlockBuffer (d3d9_device.cpp:5635-5661): D3D_OK always.
template <class Stub, class Com, int I, class Desc>
HRESULT STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::Unlock() {
    if (lock_count_ && --lock_count_ != 0) return answer(I, 12, D3D_OK);
    if (dirty_hi_ > dirty_lo_ && gpu_ && this->dev_) {
        if (this->dev_->gpu()->upload_buffer(gpu_, dirty_lo_, store_.data() + dirty_lo_, dirty_hi_ - dirty_lo_))
            this->dev_->buffer_uploaded(dirty_hi_ - dirty_lo_);
        else this->dev_->backend_failure("buffer upload", this->dev_->gpu()->error());
    }
    dirty_lo_ = dirty_hi_ = 0;
    return answer(I, 12, D3D_OK);
}

template <class Stub, class Com, int I, class Desc>
HRESULT STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::GetDesc(Desc* d) {
    if (!d) return answer(I, 13, D3DERR_INVALIDCALL);
    std::memset(d, 0, sizeof *d);
    d->Format = I == I_VB ? D3DFMT_VERTEXDATA : D3DFORMAT(fmt_);
    d->Type = I == I_VB ? D3DRTYPE_VERTEXBUFFER : D3DRTYPE_INDEXBUFFER;
    d->Usage = usage_;
    d->Pool = D3DPOOL(pool_);
    d->Size = len_;
    if constexpr (I == I_VB) d->FVF = fmt_;
    return answer(I, 13, D3D_OK);
}

template class BufferBase<D9gStub_IDirect3DVertexBuffer9, IDirect3DVertexBuffer9, I_VB, D3DVERTEXBUFFER_DESC>;
template class BufferBase<D9gStub_IDirect3DIndexBuffer9, IDirect3DIndexBuffer9, I_IB, D3DINDEXBUFFER_DESC>;

// ------------------------------------------------------------------ declarations
VertexDecl::VertexDecl(Device* d, std::vector<D3DVERTEXELEMENT9> elements, DWORD fvf)
    : Child(d), elements_(std::move(elements)), fvf_(fvf) {
    info_ = decl_info(elements_.data(), elements_.size());
    // D3D9VertexDecl::Classify (d3d9_vertex_declaration.cpp:357-358): stream 0's size, the
    // largest Offset + GetDecltypeSize(Type) (d3d9_util.cpp), which GetUPBufferSize reads
    std::vector<VertexElem> ve;
    for (const D3DVERTEXELEMENT9& e : elements_) ve.push_back(VertexElem{e.Stream, e.Offset, e.Type, e.Method, e.Usage, e.UsageIndex});
    size0_ = decl_stream0_size(ve.data(), ve.size());
    if (dev_ && !fvf_) id_ = dev_->next_decl_id();   // CreateVertexDeclaration's order (the census's decl index)
    Report& r = report();
    r.decl_elements += elements_.size();
    r.decl_elements_without_sdl += info_.without_sdl_format;
    for (uint32_t m = info_.bgra_mask; m; m &= m - 1) r.decl_bgra_inputs++;
    if (dev_ && !fvf_) dev_->enlist(this, "vertex_declaration");   // an FVF's is the device's own
}

// ------------------------------------------------------------------ shaders
template <class Stub, class Com, int I>
Shader<Stub, Com, I>::Shader(Device* d, std::vector<uint32_t> tokens, std::shared_ptr<const shader::Translated> translated)
    : Child<Stub, Com, I>(d), tokens_(std::move(tokens)), translated_(std::move(translated)) {
    if (this->dev_) this->dev_->enlist(this, I == I_VS ? "vertex_shader" : "pixel_shader");
    if (this->dev_) id_ = this->dev_->next_shader_id(I == I_VS);
    if (this->dev_) this->dev_->note_shader_bytecode(I == I_VS, id_, tokens_, translated_->reflection);
}
template class Shader<D9gStub_IDirect3DVertexShader9, IDirect3DVertexShader9, I_VS>;
template class Shader<D9gStub_IDirect3DPixelShader9, IDirect3DPixelShader9, I_PS>;

size_t shader_tokens(const DWORD* fn, bool vertex, bool* ni, std::string* why) {
    *ni = false;
    if (!fn) { *why = "no bytecode"; return 0; }
    const uint32_t version = fn[0];
    const uint32_t kind = version >> 16, major = (version >> 8) & 0xFF, minor = version & 0xFF;
    if (kind != (vertex ? 0xFFFEu : 0xFFFFu)) { *why = "the bytecode is not of this stage"; return 0; }
    // d3d9.shaderModel = 3 (host/player/dxvk.conf); d3d9_device.cpp:7921-7943.
    if (vertex) {
        if (major > 3 || (major == 1 && minor > 1) || (major == 3 && minor != 0)) { *why = "unsupported VS version"; return 0; }
    } else {
        if (major > 3 || (major == 1 && minor > 4) || (major == 3 && minor != 0)) { *why = "unsupported PS version"; return 0; }
    }
    if (major < 2) { *ni = true; *why = "shader model 1 bytecode (no instruction lengths to walk)"; return 0; }
    // Walk the instructions to END: a comment's length is in bits 16-30, an
    // instruction's in bits 24-27 (shader model 2 and 3).
    constexpr size_t kCap = 1u << 20;
    size_t i = 1;
    while (i < kCap) {
        const uint32_t t = fn[i];
        if (t == 0x0000FFFFu) return i + 1;
        if ((t & 0xFFFFu) == 0xFFFEu) i += 1 + ((t >> 16) & 0x7FFFu);
        else i += 1 + ((t >> 24) & 0x0Fu);
    }
    *ni = true;
    *why = "no END token within 1M tokens";
    return 0;
}

}  // namespace d9g
