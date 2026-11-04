"""Adapter for Microsoft's PhotoDNA Cloud Service, for licensed organisations.

PhotoDNA is proprietary. Microsoft licenses it to vetted organisations, and
this module never computes a PhotoDNA hash itself: it sends the image to the
Cloud Service with your own subscription key and hands back the service's
verdict untouched. Applications for a licence are taken at
https://www.microsoft.com/en-us/photodna.

Credentials come only from the environment (or arguments to the client):

    SPADE_PHOTODNA_KEY        your Cloud Service subscription key
    SPADE_PHOTODNA_ENDPOINT   optional, to override the default Match URL

The key is never logged, stored, echoed, or included in an error message.

The request and response format here follows Microsoft's published PhotoDNA
Cloud Service API reference as it stood when this was written (a POST to the
Match endpoint with the key in the ``Ocp-Apim-Subscription-Key`` header and a
JSON body holding the base64 image). It must be confirmed against the
documentation that comes with a licence, and it has only been tested against a
mock: no licensed key was available during development.
"""

import base64
import logging
import os
from typing import Optional
from urllib.parse import urlsplit

import requests

__all__ = ["PhotoDnaClient", "PhotoDnaError"]

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.microsoftmoderator.com/photodna/v1.0/Match"
KEY_HEADER = "Ocp-Apim-Subscription-Key"
APPLY_URL = "https://www.microsoft.com/en-us/photodna"


class PhotoDnaError(Exception):
    """The PhotoDNA service can't be used, or it didn't give a usable answer."""


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _check_endpoint(endpoint: str) -> None:
    """Refuse endpoints that would send the key or image somewhere unsafe."""
    try:
        parts = urlsplit(endpoint)
        host = parts.hostname
    except ValueError:
        raise PhotoDnaError("The PhotoDNA endpoint isn't a valid URL") from None
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise PhotoDnaError("The PhotoDNA endpoint must not contain a user name or password")
    if parts.scheme == "https" and host:
        return
    if parts.scheme == "http" and host in _LOCAL_HOSTS:
        return
    raise PhotoDnaError("The PhotoDNA endpoint must be an https:// URL "
                        "(plain http is only allowed for localhost)")


class PhotoDnaClient:
    """Sends images to the PhotoDNA Cloud Service Match call."""

    def __init__(self, key: Optional[str] = None, endpoint: Optional[str] = None,
                 timeout: float = 30.0):
        key = key or os.environ.get("SPADE_PHOTODNA_KEY")
        if not key:
            raise PhotoDnaError(
                "PhotoDNA needs a Microsoft PhotoDNA licence and a Cloud Service key. "
                f"Microsoft takes applications at {APPLY_URL}. Once you have a key, "
                "set it in the SPADE_PHOTODNA_KEY environment variable."
            )
        endpoint = endpoint or os.environ.get("SPADE_PHOTODNA_ENDPOINT") or DEFAULT_ENDPOINT
        _check_endpoint(endpoint)
        self._key = key
        self.endpoint = endpoint
        self.timeout = timeout

    def __repr__(self):
        return f"PhotoDnaClient(endpoint={self.endpoint!r}, timeout={self.timeout!r})"

    def match(self, image_bytes: bytes) -> dict:
        """Ask the service about an image; returns its JSON verdict as-is."""
        body = {"DataRepresentation": "inline",
                "Value": base64.b64encode(image_bytes).decode("ascii")}
        try:
            # The exceptions requests raises can echo request details, so only
            # their type is reported, never their text.
            r = requests.post(self.endpoint, headers={KEY_HEADER: self._key},
                              json=body, timeout=self.timeout, allow_redirects=False)
        except requests.Timeout:
            log.warning("PhotoDNA request timed out after %ss", self.timeout)
            raise PhotoDnaError(f"PhotoDNA service didn't answer within {self.timeout:g} seconds") from None
        except requests.RequestException as e:
            log.warning("PhotoDNA request failed (%s)", type(e).__name__)
            raise PhotoDnaError(f"Couldn't reach the PhotoDNA service ({type(e).__name__})") from None

        status = r.status_code
        try:
            verdict = r.json()
        except ValueError:
            verdict = None
        if 300 <= status < 400:
            # Following a redirect could forward the key and image to another host.
            log.warning("PhotoDNA service returned a redirect (HTTP %s)", status)
            raise PhotoDnaError(f"PhotoDNA service answered with a redirect (HTTP {status}); "
                                "check the endpoint") from None
        if status >= 400:
            log.warning("PhotoDNA service returned HTTP %s", status)
            hint = " (check the subscription key)" if status in (401, 403) else ""
            raise PhotoDnaError(f"PhotoDNA service returned HTTP {status}{hint}") from None
        if not isinstance(verdict, dict):
            log.warning("PhotoDNA service returned something that isn't JSON")
            raise PhotoDnaError("PhotoDNA service returned a response that isn't JSON "
                                "(an error page, perhaps); check the endpoint") from None
        return verdict
