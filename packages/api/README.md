# cambotix-ocr-api

REST API (FastAPI) for the [cambotix-ocr](https://pypi.org/project/cambotix-ocr/)
engine, implementing the versioned `/v1` contract described in the
[API reference](https://github.com/Alujack/cambotix.ocr/blob/master/docs/api-reference.md).
Clients: [TypeScript](https://github.com/Alujack/cambotix.ocr/tree/master/packages/sdk-typescript)
and [Dart/Flutter](https://github.com/Alujack/cambotix.ocr/tree/master/packages/sdk-flutter).

```bash
pip install cambotix-ocr-api
khmerocr serve --host 127.0.0.1 --port 8000 --model khmerocr-id-v1
```

Before other machines can reach it, set `KHMEROCR_API_KEY` and serve over
HTTPS. **Experimental:** every result must be reviewed by a person.
