# Nexa Care AI — Streamlit deployment

This package preserves the latest black-and-blue Nexa interface, transparent car assets, reversible hotspots, services, brands, conversation controls, and original server-side automotive advisor. Streamlit owns each visitor’s conversation; the frontend communicates through an official Streamlit component instead of a separate local HTTP server.

## Deploy on Streamlit Community Cloud

Upload this directory to a GitHub repository, excluding everything in `.gitignore`. Create an app in Streamlit Community Cloud using that repository and `app.py` as the entrypoint. Select Python 3.12. The frontend bundle is already built; no Node build is required in the cloud.

In the app’s Advanced settings → Secrets, add `OPENAI_API_KEY` privately. Never commit it or paste it in chat. Optional settings are `OPENAI_MODEL` (defaults to the existing `gpt-5.6`) and `PORTFOLIO_URL` (defaults to the existing portfolio Site). The API key never reaches the browser. Live replies remain visibly unconfigured until the key is present.

After successful deployment, use the actual returned Streamlit URL for the portfolio’s fourth-page link. No Streamlit URL has been assumed or invented in this package.

## Run locally

Install `requirements.txt`, then run `streamlit run app.py`. Configure the API key in a private environment variable or `.streamlit/secrets.toml` when live replies are required.

To rebuild the frontend after editing the bridge, run `npm install` then `npm run build`. Frontend assets are already present under `frontend/`.

Run `python -m unittest test_chatbot_guardrails -v` for the original mocked advisor checks.
