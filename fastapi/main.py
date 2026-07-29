
from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException




class Priority(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

    

class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description='Name of todo')
    todo_description: str = Field(..., min_length=3, max_length=512, description='Description of todo')
    priority: Priority = Field(..., description='Priority of todo')


class TodoCreate(TodoBase):
    pass


class Todo(TodoBase):
    todo_id: int = Field(..., description='Unique identifier of Todo')

class UpdateTodo(TodoBase):
    todo_name: Optional[str] = Field(..., min_length=3, max_length=512, description='Name of todo')
    todo_description: Optional[str] = Field(..., min_length=3, max_length=512, description='Description of todo')
    priority: Optional[Priority] = Field(..., description='Priority of todo')


all_todos = [
    Todo(todo_id=1, todo_name="Sports", todo_description="Go to gym", priority=Priority.MEDIUM),
    Todo(todo_id=2, todo_name="Read", todo_description="Read 10 pages", priority=Priority.MEDIUM),
    Todo(todo_id=3, todo_name="Shop", todo_description="Go shopping", priority=Priority.MEDIUM),
    Todo(todo_id=4, todo_name="Study", todo_description="Study for exam", priority=Priority.MEDIUM),
    Todo(todo_id=5, todo_name="Meditate", todo_description="Meditate for 20 minutes", priority=Priority.MEDIUM)
]


app = FastAPI()


@app.get('/')
def index():
    return {'message': 'hello world'}



@app.get('/todos', response_model=List[Todo])
def limit_view(first_n: int | None = None):
    if first_n:
        return all_todos[:first_n]
    else:
        return all_todos


@app.get('/todos/{todo_id}')
def get_todo(todo_id: int):
    for todo in all_todos:
        if todo.todo_id == todo_id: 
            return todo
        
    raise HTTPException(status_code=404, detail="Item not found")



@app.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id: int, updated_todo: UpdateTodo):
    for todo in all_todos:
        if todo_id == todo.todo_id:
            todo.todo_name = updated_todo.todo_name
            todo.todo_description = updated_todo.todo_description
            todo.priority = updated_todo.priority
        
            return todo
    
@app.post('/todos/{todo_id}', response_model=Todo)
def create_todo(new_todo: TodoCreate):
    new_todo_id = max(todo.todo_id for todo in all_todos) + 1

    new_todo_entry = Todo(
        todo_id=new_todo_id,
        todo_name=new_todo.todo_name,
        todo_description=new_todo.todo_description,
        priority=new_todo.priority
    )

    all_todos.append(new_todo_entry)

    return new_todo_entry


@app.delete('/todos/{todo_id}', response_model=dict)
def delete_todo(todo_id: int):

    for index, todo in enumerate(all_todos):
        if todo.todo_id == todo_id:
            popped_todo = all_todos.pop(index)
            return {'messsage': f'Popped Todo ID: {popped_todo}'}

            