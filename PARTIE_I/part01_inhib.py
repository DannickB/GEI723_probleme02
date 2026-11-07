from brian2 import *
import matplotlib.pyplot as plt

T = 100*second
N = 1000
F = 6*Hz

# Param Neurones
w_inh = 0.05
Ee = 0*mV
Ei = -80*mV
taue = 5*ms
taui = 10*ms

El = -74*mV
vt = -54*mV
vr = -60*mV
taum = 10*ms
taupre = 20*ms
taupost = taupre

gmax = .01
F_ref = 15*Hz
eta = 0.01 * gmax * (F_ref / F)
dApre, dApost = eta, -1.05 * eta

tau_theta = 5*second
dtheta = 0.1*mV

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + gi * (Ei - v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
dgi/dt = -gi / taui : 1
dtheta/dt = -theta / tau_theta : volt
taum: second (constant)
'''

# Neurones -----------------------------------------------------------------------------

poisson_input = PoissonGroup(N, rates=F)
E = NeuronGroup(1, eqs_neurons, threshold='v>vt + theta', reset='v = vr; theta += dtheta', method='euler')
I = NeuronGroup(1, eqs_neurons, threshold='v>vt + theta', reset='v = vr; theta += dtheta', method='euler')

E.v = vr
I.v = vr
E.taum = 5*ms
I.taum = 5*ms

# Connection Synapse -------------------------------------------------------------------

def graphique_choisi(forme):
    match forme:
        case 0:
            return ('''
            w : 1
            dApre/dt  = -Apre  / taupre  : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            ''',
            '''
            ge += w
            Apre += dApre
            w = clip(w + Apost * w / gmax, 0, gmax)
            ''',
            '''
            Apost += dApost
             w = clip(w + Apre * (1 - w / gmax), 0, gmax)
             ''')
        case 1:
            return ('''
            w : 1
            dApre/dt  = (0.0005-Apre)  / taupre  : 1 (event-driven)
            dApost/dt = (0.0005-Apost) / taupost : 1 (event-driven)
            ''',
            '''
            ge += w
            Apre -= dApre
            w = clip(w + Apost, 0, gmax)
            ''',
            '''
            Apost -= dApost
            w = clip(w + Apre, 0, gmax)
            ''')
        case 2:
            return ('''
            w : 1
            dApre/dt  = -Apre  / taupre  : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            ''',
            '''
            ge += w
            Apre -= dApre
            w = clip(w + Apost, 0, gmax)
            ''',
            '''
            ''')
    return None

stdp_model, stdp_pre, stdp_post = graphique_choisi(0)

def stdp_synapses(source, target, forme):
    S = Synapses(source, target, stdp_model,
                 on_pre=stdp_pre, on_post=stdp_post,
                 namespace={'dApre': forme * dApre, 'dApost': forme * dApost,
                            'taupre': taupre, 'taupost': taupost,
                            'gmax': gmax})
    S.connect()  # tout-à-tout : 1000 → 1
    S.w = 'rand() * gmax'
    return S


S_EE = stdp_synapses(poisson_input, E, 1)
S_EI = stdp_synapses(poisson_input, I, 1)

# Synapse inhibitrice fixe I → E
S_IE = Synapses(I, E, on_pre='gi += w_inh')
S_IE.connect()

# ------------------------------------------------------------------
# Moniteurs
# ------------------------------------------------------------------
rec = range(0, N, 100)  # 10 synapses
mon_EE = StateMonitor(S_EE, 'w', record=rec, dt=100 * ms)
mon_EI = StateMonitor(S_EI, 'w', record=rec, dt=100 * ms)
spk_E = SpikeMonitor(E)
spk_I = SpikeMonitor(I)

run(T, report='text')

print(f"Taux moyen E : {spk_E.num_spikes / (T / second):.1f} Hz")
print(f"Taux moyen I : {spk_I.num_spikes / (T / second):.1f} Hz")

# ------------------------------------------------------------------
# Graphiques
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(11, 9))

ax[0, 0].hist(S_EE.w / gmax, 20, color='tab:blue')
ax[0, 0].set_title(f'E→E (exc) : poids finaux')
ax[0, 1].hist(S_EI.w / gmax, 20, color='tab:red')
ax[0, 1].set_title(f'E→I (inhib) : poids finaux')
for a_ in ax[0]:
    a_.set_xlabel('w / gmax')

ax[1, 0].plot(mon_EE.t / second, mon_EE.w.T / gmax, lw=0.8)
ax[1, 0].set_title('E→E : évolution de 10 poids')
ax[1, 1].plot(mon_EI.t / second, mon_EI.w.T / gmax, lw=0.8)
ax[1, 1].set_title('E→I : évolution de 10 poids')
for a_ in ax[1]:
    a_.set_xlabel('Temps (s)')
    a_.set_ylabel('w / gmax')

bins = np.arange(0, T / second + 1, 1)
for spk, nom, c in [(spk_E, 'E', 'tab:blue'), (spk_I, 'I', 'tab:red')]:
    taux, _ = np.histogram(spk.t / second, bins=bins)
    ax[2, 0].plot(bins[:-1], taux, label=nom, color=c)
ax[2, 0].set_title('Taux de décharge (fenêtres de 1 s)')
ax[2, 0].set_xlabel('Temps (s)')
ax[2, 0].set_ylabel('Hz')
ax[2, 0].legend()

ax[2, 1].plot(mon_EE.t / second, mon_EE.w.mean(axis=0) / gmax, label='E→E',
              color='tab:blue')
ax[2, 1].plot(mon_EI.t / second, mon_EI.w.mean(axis=0) / gmax, label='E→I',
              color='tab:red')
ax[2, 1].set_title('Poids moyen (10 synapses)')
ax[2, 1].set_xlabel('Temps (s)')
ax[2, 1].legend()

plt.tight_layout()
plt.show()