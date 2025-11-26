import os
import subprocess
import sys

# ============================
# --- Liste des dépendances ---
# ============================

SYSTEM_PACKAGES = [
    "python3-pip",
    "python3-rpi.gpio",
    "python3-gpiozero"
]

PYTHON_PACKAGES = [
    "RPi.GPIO",
    "gpiozero",
    "numpy",
    "pyrplidar"
]

# ============================
# --- Fonctions utilitaires ---
# ============================

def run_command(command, use_sudo=False):
    """Exécute une commande shell avec affichage."""
    try:
        cmd = command
        if use_sudo and os.geteuid() != 0:
            cmd.insert(0, "sudo")
        print(f"🔧 Exécution : {' '.join(cmd)}")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Erreur lors de l'exécution : {e}")
    except Exception as ex:
        print(f"❌ Erreur inattendue : {ex}")

def install_system_packages():
    """Installe les paquets système nécessaires (apt)."""
    print("\n📦 Installation des dépendances système...")
    run_command(["apt", "update"], use_sudo=True)
    for pkg in SYSTEM_PACKAGES:
        run_command(["apt", "install", "-y", pkg], use_sudo=True)
    print("✅ Dépendances système installées.\n")

def install_python_packages():
    """Installe les paquets Python nécessaires (pip)."""
    print("🐍 Installation des dépendances Python...")
    for pkg in PYTHON_PACKAGES:
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
    print("✅ Dépendances Python installées.\n")

# ============================
# --- Programme principal ---
# ============================

def main():
    print("🚀 Installation complète des dépendances pour la voiture autonome...")
    install_system_packages()
    install_python_packages()
    print("🎉 Installation terminée avec succès ! Votre environnement est prêt à être utilisé.\n")

if __name__ == "__main__":
    main()
