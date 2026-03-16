FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages: build toolchain, IM7 delegates, exiftool ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config git ca-certificates \
        libtiff-dev liblcms2-dev \
        libjpeg-dev libpng-dev libwebp-dev libgif-dev \
        libfreetype6-dev libxml2-dev libfontconfig1-dev \
        libheif-dev libde265-dev libopenjp2-7-dev \
        libbz2-dev zlib1g-dev libltdl-dev \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# ── ImageMagick 7 from source ────────────────────────────────────────
RUN git clone --depth 1 https://github.com/ImageMagick/ImageMagick.git /tmp/imagemagick \
    && cd /tmp/imagemagick \
    && ./configure \
        --prefix=/usr/local \
        --with-quantum-depth=16 \
        --enable-hdri \
        --with-tiff=yes \
        --with-lcms=yes \
        --with-jpeg=yes \
        --with-png=yes \
        --with-webp=yes \
        --with-xml=yes \
        --enable-shared \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/imagemagick

# ── Python + pip dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir --break-system-packages \
        pyvips \
        numpy \
        paddlepaddle==3.2.2 \
        paddleocr

# ── Application repository ───────────────────────────────────────────
RUN git clone --depth 1 https://github.com/nathanvercaemert/IMAGE_PROCESSING.git /opt/image_processing

WORKDIR /opt/image_processing

# ── Verify installs ─────────────────────────────────────────────────
RUN magick -version \
    && magick -list configure | grep DELEGATES \
    && exiftool -ver
