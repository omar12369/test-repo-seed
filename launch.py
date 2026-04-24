# launch.py
if __name__ == "__main__":
    import uvicorn

    # NOTE: import string form so --reload works
    uvicorn.run("runtime.main:app", host="127.0.0.1", port=8000, reload=True)
