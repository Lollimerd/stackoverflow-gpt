import os
from pathlib import Path

def get_size(path):
    """Return total size of a file or directory in bytes"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_size(entry.path)
    return total

def format_size(size):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def print_tree(directory, prefix='', level=-1, show_hidden=False):
    """Print directory tree with sizes"""
    if level == 0:
        return
    
    files = []
    dirs = []
    
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not show_hidden and entry.name.startswith('.'):
                    continue
                if entry.is_dir():
                    dirs.append(entry)
                else:
                    files.append(entry)
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return
    
    entries = dirs + files
    
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        size = get_size(entry.path)
        
        if entry.is_dir():
            print(f"{prefix}{'└── ' if is_last else '├── '}📁 {entry.name}/ ({format_size(size)})")
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}📄 {entry.name} ({format_size(size)})")
        
        if entry.is_dir():
            extension = '    ' if is_last else '│   '
            print_tree(entry.path, prefix + extension, level - 1 if level > 0 else level, show_hidden)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Display folder tree with sizes')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to scan')
    parser.add_argument('-l', '--level', type=int, default=-1, help='Depth level to display (-1 for unlimited)')
    parser.add_argument('-a', '--all', action='store_true', help='Show hidden files and directories')
    
    args = parser.parse_args()
    
    root_dir = Path(args.directory).resolve()
    print(f"{root_dir}/ ({format_size(get_size(root_dir))})")
    print_tree(root_dir, level=args.level, show_hidden=args.all)