"""Tests for DROCAT's dependency-light NeuronBridge API client."""

from io import BytesIO

from PIL import Image

from neuronbridge_client import Client


class FakeResponse:
    def __init__(self, *, text="", payload=None, content=b"", status=200):
        self.text = text
        self._payload = payload
        self.content = content
        self.raw = BytesIO(content)
        self.status_code = status
        self.closed = False

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return self.responses[url]


def _client_and_urls():
    root = "https://test-bucket.s3.us-east-1.amazonaws.com"
    version_root = f"{root}/v-test"
    config = {
        "defaultSearchLibrary": "ignored-new-api-field",
        "stores": {
            "brain": {
                "prefixes": {
                    "CDSResults": f"{version_root}/matches/",
                    "CDM": f"{version_root}/images/",
                }
            }
        },
    }
    responses = {
        f"{root}/current.txt": FakeResponse(text="v-test\n"),
        f"{version_root}/config.json": FakeResponse(payload=config),
    }
    session = FakeSession(responses)
    return Client(data_bucket="test-bucket", session=session), session, responses, version_root


def test_client_ignores_new_config_fields_and_builds_nested_objects():
    client, _session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_body/42.json"
    responses[lookup_url] = FakeResponse(payload={
        "results": [{
            "type": "EMImage",
            "id": "em-42",
            "publishedName": "hemibrain:v1.2.1:42",
            "files": {"store": "brain", "CDSResults": "42.json"},
        }]
    })

    images = client.get_em_images(42)

    assert client.version == "v-test"
    assert client.config.defaultSearchLibrary == "ignored-new-api-field"
    assert images[0].publishedName == "hemibrain:v1.2.1:42"
    assert images[0].files.CDSResults == "42.json"
    assert all(response.closed for response in responses.values())


def test_client_fetches_matches_and_resolves_store_prefixes():
    client, session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_line/LH173.json"
    match_url = f"{version_root}/matches/lh173.json"
    responses[lookup_url] = FakeResponse(payload={
        "results": [{
            "type": "LMImage",
            "id": "lm-1",
            "files": {"store": "brain", "CDSResults": "lh173.json"},
        }]
    })
    responses[match_url] = FakeResponse(payload={
        "results": [{
            "type": "CDSMatch",
            "normalizedScore": 0.91,
            "files": {"store": "brain"},
            "image": {"type": "EMImage", "id": "em-1", "files": {"store": "brain"}},
        }]
    })

    matches = client.get_cds_matches(client.get_lm_images("LH173")[0])

    assert match_url in session.urls
    assert matches[0].normalizedScore == 0.91
    assert matches[0].image.id == "em-1"


def test_client_loads_images_from_match_or_neuron_files():
    client, _session, responses, version_root = _client_and_urls()
    image_url = f"{version_root}/images/sample.png"
    buffer = BytesIO()
    Image.new("RGB", (2, 3), "red").save(buffer, format="PNG")
    responses[image_url] = FakeResponse(content=buffer.getvalue())
    match = type("Match", (), {
        "files": type("Files", (), {"store": "brain", "CDM": None})(),
        "image": type("ImageObject", (), {
            "files": type("Files", (), {"store": "brain", "CDM": "sample.png"})()
        })(),
    })()

    image = client.get_cds_image(match)

    assert image.size == (2, 3)
    assert responses[image_url].closed


def test_client_url_encodes_lookup_ids():
    client, session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_line/line%2Fwith%20space.json"
    responses[lookup_url] = FakeResponse(payload={"results": []})

    assert client.get_lm_images("line/with space") == []
    assert session.urls[-1] == lookup_url
