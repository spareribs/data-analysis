#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/13 16:13
# @Author  : Spareribs
# @File    : aa.py
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import time
from IPy import IP


def galaxy(ip):
    URL = "http://192.168.229.197:8088/api/v1/machines/admin_action"
    token = 'MUNGedi9Cjki%2B6lMyzOeLFBtAhYfm38kbxk2vDjjZhIdPD13n25pnOqtMi45EhLXPubsWwQkANtM%0Atzt765WsoE%2FDlWJfuuutrf48u8DKzhp3wnMJ6Sq0MPYCSkseBFwUIKlsSTD0SevceHv0SVuaPP20%0Ag27A1L6vOCOh08ip1F0%3D%0A'
    headers = {'Content-Type': 'application/json',
               'X-Auth-Token': token,
               'X-Auth-Project': 'test'}
    payload = {"action": "offline", "ips": [ip]}
    # r = requests.get(url2, headers=headers)
    res = requests.post(URL, headers=headers, json=payload)
    print(res.status_code)


if __name__ == '__main__':
    all_ip = IP('42.186.73.0/24')
    for ip in all_ip:
        print(ip)
        print([str(ip)])
        galaxy([str(ip)])
        time.sleep(3)

    for ip_tag in range(0, 255):
        galaxy(["42.186.73.{0}".format(ip_tag)])
        time.sleep(3)
