#
# Borrowing much from MediaCloud:
# https://github.com/berkmancenter/mediacloud/blob/master/apps/base/Dockerfile
#

FROM nvidia/cuda:10.2-base

ENV DEBIAN_FRONTEND=noninteractive \
    # don't send stdout to logs
    PYTHONUNBUFFERED=1 \
    # don't build .pyc files
    PYTHONDONTWRITEBYTECODE=1

# upgrade packages
RUN \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y autoremove && \
    apt-get -y clean

# install base dependencies
RUN \
    apt-get -y --no-install-recommends install apt-utils && \
    apt-get -y --no-install-recommends install apt-transport-https && \
    apt-get -y --no-install-recommends install acl && \
    apt-get -y --no-install-recommends install sudo && \
    apt-get -y --no-install-recommends install file && \
    true

# install common packages
RUN \
    apt-get -y --no-install-recommends install \
        # Quicker container debugging
        bash-completion \
        # Some of the packages might want to use SSL
        ca-certificates \
        curl \
        htop \
        # "ip" and similar utilities
        iproute2 \
        # Pinging other containers from within Compose environment
        iputils-ping \
        # Sending mail via sendmail utility through mail-postfix-server
        msmtp \
        msmtp-mta \
        # Provides killall among other utilities
        psmisc \
        less \
        locales \
        # Waiting for some port to open
        netcat \
        # Some packages insist on logging to syslog
        rsyslog \
        # "mail" utility (which uses msmtp internally)
        s-nail \
        # Timezone data, used by many packages
        tzdata \
        # Basic editor for files in container while debugging
        # (full vim doesn't cut it because it's too big and also installs
        # Python 3.5 which might interfere with a newer Python 3.7 that we use
        # for our app)
        vim-tiny \
    && \
    true

# install python and pip
RUN \
    apt-get -y --no-install-recommends install \
        build-essential \
        python3.7 \
        python3.7-dev \
        python3-pip \
        python3-setuptools \
        python3-distutils \
        python3-apt \
        python3-wheel \
    && \
    # make python available as python3
    # ln -s /usr/bin/python3.7 /usr/bin/python3 && \
    # ln -s /usr/lib/python3.7 /usr/lib/python3 && \
    # install pip
    # curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.7 && \
    # rm -rf /root/.cache/ && \
    true

# fix locale settings
RUN \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure locales && \
    true

# install CUDA


# install python packages
COPY requirements.txt /var/tmp/
RUN \
    cd /var/tmp && \
    pip3 install -r requirements.txt && \
    rm requirements.txt && \
    rm -rf /root/.cache/ && \
    true

COPY . sdemb/

# run jupyter
# CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

# docker run -it -p 8888:8888 test:v1
