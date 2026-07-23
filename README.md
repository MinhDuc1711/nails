# NAILS
Algorithm to identify color variations from fingernails using A.I.

We present an algorithm that allows to identify color variations from fingernails using Artificial Intelligence (A.I.). Healthy nails have a uniform color and are visually smooth. However, as one ages, nails may become more brittle and may have discoloration due to injury, fungal and viral infections, medications, etc. Furthermore, the appearance of the fingernails can also change due to some medical conditions. 

#### User Manual

User Manual for install and usage: https://github.com/canessae/nails/blob/main/NAILS-UserManual-v1.pdf

#### NAILS GUI

<picture>
  <img alt="NAILS GUI" src="https://github.com/canessae/nails/blob/main/GraphicalAbstract.png" width="550">
</picture>

## Run the web app

Install the web dependencies:

```bash
python3.10 -m pip install -r requirements-web.txt
```

Start the site locally:

```bash
python3.10 web_app.py
```

Or run it with Gunicorn:

```bash
gunicorn --bind 0.0.0.0:$PORT wsgi:app
```

For deployment to a hosting service such as Render, Railway, or Heroku, make sure the environment exposes a `PORT` value and use the Gunicorn entry point from the included Procfile.

#### Citation

If this software (including any part of it) is useful for you, please cite

```bibtex
@misc{
  title = {NAILS: Normalized Artificial Intelligence Labeling Sensor for Self-Care Health},
  author = {L. Tenze, E. Canessa},
  journal = {Sensors 2024, 24(24), 7997},
  doi = {10.3390/s24247997},
  url = {https://www.mdpi.com/1424-8220/24/24/7997},
}
```
