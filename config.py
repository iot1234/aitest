# gunicorn.conf.py - TIMEOUT OPTIMIZED CONFIGURATION
import os
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # ลดจำนวน workers เพื่อลดการใช้ memory
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings - เพิ่ม timeout เพื่อรองรับการฝึกโมเดล
timeout = int(os.environ.get('WORKER_TIMEOUT', '60'))  # เพิ่มเป็น 60 วินาที
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ai_prediction_system'

# Server mechanics
preload_app = True
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None

# Worker process management
max_worker_memory = 512 * 1024 * 1024  # 512MB per worker
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance

# Application specific
raw_env = [
    f'TRAINING_TIMEOUT={int(os.environ.get("TRAINING_TIMEOUT", "50"))}',  # 50 วินาที สำหรับการฝึกโมเดล
    f'WORKER_TIMEOUT={timeout}',
]

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("🚀 AI Prediction System is ready to serve requests")
    server.log.info(f"⏰ Worker timeout: {timeout} seconds")
    server.log.info(f"👥 Workers: {workers}")

def worker_int(worker):
    """Called just after a worker has been killed by a signal."""
    worker.log.info(f"🔄 Worker {worker.pid} received INT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"🔧 Forking worker {worker.age}")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"✅ Worker {worker.pid} spawned")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info(f"❌ Worker {worker.pid} aborted")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("🔄 Forked parent, pre-execution")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug(f"📥 {req.method} {req.path}")

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug(f"📤 {req.method} {req.path} - {resp.status}")

def child_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"👋 Worker {worker.pid} exited")

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"🚪 Worker {worker.pid} exit")

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info(f"🔢 Number of workers changed from {old_value} to {new_value}")

def on_exit(server):
    """Called just before exiting."""
    server.log.info("👋 AI Prediction System is shutting down")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("🔄 Reloading AI Prediction System")

# Error handling
def worker_timeout(worker):
    """Called when a worker times out."""
    worker.log.error(f"⏰ Worker {worker.pid} timed out after {timeout} seconds")
    worker.log.error("💡 Consider increasing WORKER_TIMEOUT or optimizing your model training")
    
# Memory management
def worker_memory_limit():
    """Return the memory limit for workers in bytes."""
    return max_worker_memory

# Custom configuration for AI workloads
def optimize_for_ai():
    """Apply AI-specific optimizations."""
    import gc
    import threading
    
    # Enable garbage collection optimization
    gc.set_threshold(700, 10, 10)
    
    # Set thread limits for scikit-learn
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Apply optimizations
optimize_for_ai()

print("🔧 Gunicorn configuration loaded with AI optimizations")
print(f"⏰ Worker timeout: {timeout} seconds")
print(f"👥 Workers: {workers}")
print(f"🧠 Memory limit per worker: {max_worker_memory // (1024*1024)}MB")

