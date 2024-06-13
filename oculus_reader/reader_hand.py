from FPS_counter import FPSCounter

import numpy as np
import threading
import time
import os
from ppadb.client import Client as AdbClient # 实现通信的核心组件
import sys

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

class OculusHandReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.DefaultCompany.hand',
            print_FPS=False,
            run=True
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = 'rightLocalWorldMatrix'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    
    def run(self):
        # 将实例的running属性设置为True，表示该方法正在运行
        self.running = True
        # 使用adb命令启动Android设备上的特定应用
        self.device.shell('am start -n com.DefaultCompany.hand/com.unity3d.player.UnityPlayerActivity')
        """
            这行代码通过adb shell命令启动Android设备上的特定应用。
            am start -n：使用Activity Manager (am)来启动一个新的Activity。
            "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity"：指定要启动的Activity的完全限定名称，com.rail.oculus.teleop是包名，com.rail.oculus.teleop.MainActivity是主Activity。
            -a android.intent.action.MAIN：指定要启动的Activity的动作，这里是主动作（MAIN）。
            -c android.intent.category.LAUNCHER：指定要启动的Activity的类别，这里是启动器类别（LAUNCHER）。
        """
        # 创建一个新线程，该线程的目标函数是self.device.shell，传递的参数是("logcat -T 0", self.read_logcat_by_line)，即在设备上运行 logcat -T 0 并使用self.read_logcat_by_line作为回调函数，用于逐行处理读取到的日志。
        self.thread = threading.Thread(target=self.device.shell, args=("logcat Unity:D *:S", self.read_logcat_by_line)) 
        # 启动该线程
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)
    
    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusHandReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)
    
    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'hand.apk')
                    print("using default APK_path:{}".format(APK_path))
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)
    
    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def process_data(self, data):
        ret = {
                        "right":data.split("&")[0],
                        "right_pinch":data.split("&")[1]
            }
        
        right = ret["right"].split(" ")

        right = np.array([float(i) for i in right])

        right = right.reshape(4,-1)

        ret["right"] = right
        
        return ret["right"], ret["right_pinch"]
    
    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = self.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()
    
    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons
    
def main():
    oculus_reader = OculusHandReader()

    while True:
        time.sleep(0.3)
        ret = oculus_reader.get_transformations_and_buttons()
        print(ret)

if __name__ == '__main__':
    main()