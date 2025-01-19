MN914 Joint Project
------

This repo is intended for storing our code for the joint project.

## Setup

### Requirements

See [`requirements.txt`](requirements.txt).

```cmd
pip install -r requirements.txt
```

###### Recommendations

It is highly recommended to create a virtual environment in the project root. For example:

```cmd
python -m venv .venv
```

Then activate it with the following command:

- For Windows:
```cmd
.\.venv\Scripts\activate.ps1
```
- For Linux:
```cmd
./.venv/Scripts/activate.bat
```

## Project Structure

```
mn914/
│   README.md
│   requirements.txt
└───.venv/               <== (optional) virtual environment, you have to create it yourself
└───hidden/              <== HiDDeN submodule
└───stable_signature/    <== Stable Signature submodule
└───resources/           <== where we store resources (figures, etc.)
```
