import fastapi
import os
import aiofiles
import base64
import time
import asyncio
from contextlib import asynccontextmanager
from pydantic import BaseModel

assigned = {}

timetolive = int(os.getenv("TIMETOLIVE", "600"))  # Default 10 minutes

all = set([x.removesuffix(".png") for x in os.listdir(os.path.join("dataset", "images"))])
done = set([x.removesuffix(".txt") for x in os.listdir(os.path.join("dataset", "ocrtext_easy"))])
undone = sorted(list(all - done))

async def cleanup_stale_assignments():
    """Background task to periodically check and remove stale assignments"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        current_time = time.time()
        stale_items = []

        # Find all stale assignments
        for filename, assigned_time in assigned.items():
            if current_time - assigned_time > timetolive:
                stale_items.append(filename)

        # Move stale items back to undone
        for filename in stale_items:
            assigned.pop(filename)
            undone.append(filename)
            print(f"Reassigned stale item: {filename}")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: create the cleanup task
    task = asyncio.create_task(cleanup_stale_assignments())
    yield
    # Shutdown: cancel the cleanup task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = fastapi.FastAPI(lifespan=lifespan)


class Work(BaseModel):
    filename: str
    text: str = None


@app.get("/get_work")
async def get_work():
    if len(undone) > 0:
        file_name = undone.pop(0)
        file_path = os.path.join("dataset", "images", file_name + '.png')
        if os.path.exists(file_path):
            assigned[file_name] = time.time()
            async with aiofiles.open(file_path, "rb") as file:
                img = await file.read()
                print(f"Assigned {file_name}")
            return {"file_name": file_name, "image": base64.b64encode(img).decode("utf-8")}
        else:
            print(f"Error: {file_path} not found")
            return {"status": "error", "message": "File not found"}
    print("No work available")
    return {"status": "no_work_available"}


@app.post("/submit_work")
async def submit_work(work: Work):
    if work.filename in assigned:
        assigned.pop(work.filename)
        file_name = work.filename + ".txt"
        if work.text is not None:
            async with aiofiles.open(os.path.join("dataset", "ocrtext_easy", file_name), "w+") as file:
                await file.write(work.text)
            done.add(work.filename)
            print(f"Wrote {file_name}")
            return True
        else:
            print(f"Error: {work.filename} returned empty")
            undone.append(work.filename)
    print(f"Error: {work.filename} not in assigned")
    return False

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "undone_count": len(undone),
        "assigned_count": len(assigned),
        "done_count": len(done)
    }