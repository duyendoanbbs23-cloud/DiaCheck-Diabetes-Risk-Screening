# DiaCheck: enable the expanded language list

DiaCheck keeps built-in language packs for English, Vietnamese, French, Simplified Chinese, Japanese, Korean, and Spanish. To offer the wider set of target languages supported by Google Cloud Translation, configure the Translation Basic API.

> This does not mean every language spoken on Earth is supported. The app retrieves the provider's current supported-language list at runtime. Language availability can change, and machine translation is not a substitute for medical translation review.

## 1. Create and configure the Google Cloud project

1. Open Google Cloud Console and create or select a project.
2. Enable **Cloud Translation API** for that project.
3. Enable billing for the project and review the current Cloud Translation pricing and quotas.
4. Create an API key. Restrict the key to the Cloud Translation API and apply suitable usage restrictions where supported.

## 2. Add the key to Streamlit Cloud

In your Streamlit Community Cloud app:

1. Open the app's **Settings**.
2. Open **Secrets**.
3. Add the following entry, replacing the placeholder with your real key:

```toml
GOOGLE_TRANSLATE_API_KEY = "YOUR_REAL_API_KEY"
```

Do not paste the real key into `app.py`, commit it to GitHub, or send it in a chat. The `.gitignore` file excludes the local `.streamlit/secrets.toml` file.

## 3. Run locally

For local testing, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then replace the placeholder. Alternatively, set the `GOOGLE_TRANSLATE_API_KEY` environment variable before starting Streamlit.

## 4. How the app uses translation

- The language dropdown is populated from the provider's supported target-language list.
- The app translates a fixed catalogue of interface text and caches the translated catalogue for 24 hours.
- Patient answers, height, weight, BMI, and model results are **not** sent to the translation API.
- The original seven built-in language packs override machine translations where a reviewed translation is available.
- If the API key is absent or the service is unavailable, the app falls back to its built-in languages.

## 5. Medical-language caution

Machine-translated wording can be inaccurate, especially for medical questions, warnings, and recommendations. Before presenting a newly added language to the public, have a qualified human translator or healthcare professional review the translated questionnaire, result explanations, and medical disclaimer. DiaCheck remains an educational project and must not be presented as a diagnostic or clinical decision tool.

## Files to add or replace in your GitHub repository

Upload these files while keeping the folder structure:

- Replace the root `app.py` with this updated version.
- Add `i18n.py` beside `app.py`.
- Add `assets/diacheck_medical_team.png`.
- Add `.gitignore`, `GLOBAL_LANGUAGES_SETUP.md`, and `.streamlit/secrets.toml.example`.

This is a user-interface and localization update. It does not change the trained model, so you do not need to run `train.py` again. Keep your repository's existing `models/pipeline.joblib`, `models/model_meta.json`, `data/`, and `requirements.txt` files.
