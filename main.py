from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import pdf2image
import tabula
import fitz
import os
import pandas as pd
import numpy as np
import re
from io import BytesIO
import shutil

app = FastAPI()

# TODO: env
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# PyTesseract Config Options
config = ('-l fr --oem 1 --psm 1')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # TODO: tesseract OCR
    # TODO: process with model
    
    response = {
        "bank_stmt_entries": [
            {
            "TXN_DATE": "01-11-2022",
            "TXN_DESC": "03860300144623 Class Proceeds",
            "CHEQUE_REF_NO": "NA",
            "WITHDRAWAL_AMT": "NA",
            "DEPOSIT_AMT": "12768.00",
            "BALANCE_AMT": "13285.22"
            }
        ],
    }
      
    return JSONResponse(content=response)

if __name__ == '_main_':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)