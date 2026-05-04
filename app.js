// ===== MOBILE MENU =====
const hamburger = document.getElementById('hamburger');
const mobileMenu = document.getElementById('mobileMenu');

if (hamburger) {
  hamburger.addEventListener('click', () => {
    mobileMenu.classList.toggle('open');
  });
}

function closeMobileMenu() {
  if (mobileMenu) mobileMenu.classList.remove('open');
}

// ===== TOAST =====
function showToast(msg, duration = 3000) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), duration);
}

// ===== AUTH TAB SWITCH =====
function switchTab(tab) {
  const authSection = document.getElementById('auth');
  if (authSection) authSection.scrollIntoView({ behavior: 'smooth' });

  const loginCard = document.getElementById('loginCard');
  const registerCard = document.getElementById('registerCard');

  if (tab === 'register') {
    if (loginCard) loginCard.style.opacity = '0.6';
    if (registerCard) {
      registerCard.style.opacity = '1';
      registerCard.querySelector('input')?.focus();
    }
  } else {
    if (registerCard) registerCard.style.opacity = '0.6';
    if (loginCard) {
      loginCard.style.opacity = '1';
      loginCard.querySelector('input')?.focus();
    }
    setTimeout(() => {
      if (loginCard) loginCard.style.opacity = '1';
      if (registerCard) registerCard.style.opacity = '1';
    }, 1500);
  }
}

// ===== LOGIN HANDLER =====
function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value;
  const pass = document.getElementById('loginPass').value;

  if (!email || !pass) {
    showToast('Please fill in all fields.');
    return;
  }

  showToast('Signing you in...');
  setTimeout(() => {
    window.location.href = 'classify.html';
  }, 1200);
}

// ===== REGISTER HANDLER =====
function handleRegister(e) {
  e.preventDefault();
  const name = document.getElementById('regName').value;
  const email = document.getElementById('regEmail').value;
  const pass = document.getElementById('regPass').value;

  if (!name || !email || !pass) {
    showToast('Please fill in all required fields.');
    return;
  }
  if (pass.length < 8) {
    showToast('Password must be at least 8 characters.');
    return;
  }

  showToast('Account created! Redirecting...');
  setTimeout(() => {
    window.location.href = 'classify.html';
  }, 1200);
}

// ===== PASSWORD STRENGTH =====
const regPass = document.getElementById('regPass');
if (regPass) {
  regPass.addEventListener('input', () => {
    const val = regPass.value;
    const fill = document.getElementById('strengthFill');
    const label = document.getElementById('strengthLabel');
    if (!fill || !label) return;

    let strength = 0;
    if (val.length >= 8) strength++;
    if (/[A-Z]/.test(val)) strength++;
    if (/[0-9]/.test(val)) strength++;
    if (/[^A-Za-z0-9]/.test(val)) strength++;

    const levels = [
      { pct: '0%', color: 'transparent', text: '' },
      { pct: '25%', color: '#ef4444', text: 'Weak' },
      { pct: '50%', color: '#f59e0b', text: 'Fair' },
      { pct: '75%', color: '#3b82f6', text: 'Good' },
      { pct: '100%', color: '#22c55e', text: 'Strong' },
    ];

    fill.style.width = levels[strength].pct;
    fill.style.background = levels[strength].color;
    label.textContent = levels[strength].text;
    label.style.color = levels[strength].color;
  });
}

// ===== ANIMATE BARS ON PAGE LOAD =====
window.addEventListener('load', () => {
  setTimeout(() => {
    document.querySelectorAll('.bar-fill').forEach(bar => {
      const target = bar.style.width;
      bar.style.width = '0%';
      requestAnimationFrame(() => {
        setTimeout(() => { bar.style.width = target; }, 50);
      });
    });
  }, 400);
});