import sys
import torch

def main():
    print("========== Python ==========")
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)

    print("\n========== PyTorch ==========")
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Compiled CUDA version:", torch.version.cuda)

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))

    print("\n========== Hugging Face ==========")
    try:
        import transformers
        print("transformers version:", transformers.__version__)
    except ImportError:
        print("transformers: NOT INSTALLED")

    try:
        import accelerate
        print("accelerate version:", accelerate.__version__)
    except ImportError:
        print("accelerate: NOT INSTALLED")

    print("\n========== UI ==========")
    try:
        import gradio
        print("gradio version:", gradio.__version__)
    except ImportError:
        print("gradio: NOT INSTALLED")

    print("\n========== Tokenizers ==========")
    for pkg in ["tokenizers", "sentencepiece", "tiktoken"]:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"{pkg} version:", version)
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

# 还要下载这个
# chromadb


if __name__ == "__main__":
    main()

