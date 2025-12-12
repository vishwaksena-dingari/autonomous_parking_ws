FROM osrf/ros:humble-desktop-full

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/autonomous_parking_ws/requirements.txt
RUN pip3 install --no-cache-dir --upgrade --ignore-installed --default-timeout=1000 -r /root/autonomous_parking_ws/requirements.txt

# Set up workspace
WORKDIR /root/autonomous_parking_ws
COPY src /root/autonomous_parking_ws/src

# Build the package
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Copy helper scripts
COPY train.sh /root/autonomous_parking_ws/train.sh
COPY eval_latest.sh /root/autonomous_parking_ws/eval_latest.sh
RUN chmod +x /root/autonomous_parking_ws/train.sh /root/autonomous_parking_ws/eval_latest.sh

# Source entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# CMD is empty so entrypoint sees no args and shows menu
CMD []
