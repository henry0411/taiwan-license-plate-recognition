"""Synology Chat incoming webhook."""
import json
import logging

import requests

log = logging.getLogger("notify")


class SynologyChat:
    def __init__(self, webhook_url, verify_ssl=True):
        self.url, self.verify = webhook_url, verify_ssl

    def _post(self, payload):
        r = requests.post(self.url, data={"payload": json.dumps(payload, ensure_ascii=False)},
                          timeout=15, verify=self.verify)
        try:
            body = r.json()
        except ValueError:
            body = {"success": False, "raw": r.text[:200]}
        return body.get("success", False), body

    def send(self, text, file_url=None):
        payload = {"text": text}
        if file_url:
            payload["file_url"] = file_url
        try:
            ok, body = self._post(payload)
            if not ok and file_url:
                # NAS 抓不到圖時退回純文字
                log.warning("附圖發送失敗 (%s), 改發純文字", body)
                ok, body = self._post({"text": text})
            if not ok:
                log.error("Synology Chat 發送失敗: %s", body)
            return ok
        except requests.RequestException as e:
            log.error("Synology Chat 連線錯誤: %s", e)
            return False
