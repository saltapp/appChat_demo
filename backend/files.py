from pydantic import BaseModel
from peewee import *
from playhouse.shortcuts import model_to_dict
from typing import List, Union, Optional
import time
import logging

from db import DB

import json


log = logging.getLogger(__name__)

####################
# Files DB Schema
####################
class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


class File(Model):
    id = CharField(unique=True)
    user_id = CharField()
    filename = TextField()
    meta = JSONField()
    created_at = BigIntegerField()

    class Meta:
        database = DB


class FileModel(BaseModel):
    id: str
    user_id: str
    filename: str
    meta: dict
    created_at: int  # timestamp in epoch


####################
# Forms
####################


class FileModelResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    meta: dict
    created_at: int  # timestamp in epoch


class FileForm(BaseModel):
    id: str
    filename: str
    meta: dict = {}


class FilesTable:
    def __init__(self, db):
        self.db = db
        self.db.create_tables([File])

    def insert_new_file(self, user_id: str, form_data: FileForm) -> Optional[FileModel]:
        file = FileModel(
            **{
                **form_data.model_dump(),
                "user_id": user_id,
                "created_at": int(time.time()),
            }
        )

        try:
            result = File.create(**file.model_dump())
            if result:
                return file
            else:
                return None
        except Exception as e:
            print(f"Error creating tool: {e}")
            return None

    def get_file_by_id(self, id: str) -> Optional[FileModel]:
        try:
            file = File.get(File.id == id)
            return FileModel(**model_to_dict(file))
        except:
            return None

    def get_files(self) -> List[FileModel]:
        return [FileModel(**model_to_dict(file)) for file in File.select()]

    def delete_file_by_id(self, id: str) -> bool:
        try:
            query = File.delete().where((File.id == id))
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False

    def delete_all_files(self) -> bool:
        try:
            query = File.delete()
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False


Files = FilesTable(DB)
