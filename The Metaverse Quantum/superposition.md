1. Schrödinger’s Cat in a Transparent Box

    Visual: A small virtual box sits on your real tabletop. Inside you see a semi-transparent cat model that simultaneously plays two looping animations (“alive” and “dead”) blended together.

    Interaction:

        Slider: Adjusts the probability amplitudes (e.g. more “alive” vs. more “dead”).

        Tap to Measure: On tap, the blend instantly snaps to either “alive” or “dead” according to the probabilities you set.

    Tech Tips:

        Use two animation states in Unity’s Animator, blend via the controller’s “mix” parameter.

        On tap, sample a random number vs. your amplitude value to choose which animation state to force.

2. Double-Slit Wave-Particle Demo

    Visual: A virtual “wavefront” (semi-opaque rippling plane) approaches two slits in a barrier, then fans out into an interference pattern on a back screen. Meanwhile, faint dots flicker into view across that screen.

    Interaction:

        Play/Pause: Let the wave continuously interfere.

        Measure Mode: Switch to “particle” view: dots appear one by one at random positions weighted by the interference intensity.

    Tech Tips:

        Precompute an interference intensity map (e.g. using a simple 2D sine-wave formula) and use it as a probability texture.

        In “particle” mode, spawn point sprites sampling that texture.

3. Qubit on the Bloch Sphere

    Visual: A translucent sphere floats above a marker. A glowing arrow (the qubit’s state vector) points somewhere on its surface—in general, neither purely “north” nor “south.”

    Interaction:

        Rotate Gesture: Drag to change the vector’s θ/φ angles; see it move smoothly over the sphere.

        Measure Button: Tapping “measure” causes the arrow to jump randomly to the north pole (|0⟩) or south pole (|1⟩), with probabilities cos²(θ/2) and sin²(θ/2).

    Tech Tips:

        Model the sphere in Blender with slight transparency; import into Unity.

        Control the arrow via spherical coordinates; random collapse uses Unity’s Random.value.

4. Heads-or-Tails Coin Toss as a Qubit

    Visual: A coin hovers above your real palm, but instead of showing “heads” or “tails,” each face is half-transparent, merged together in a “both” state.

    Interaction:

        Spin Gesture: Swipe to spin the coin—during the spin it remains blended.

        Tap to Measure: Stops the spin and flips the transparency instantly to 100% one face, chosen at random.

    Tech Tips:

        Use two coin-face materials with adjustable alpha.

        On tap, animate the alpha to 0/1 based on a random choice.

5. Photon Polarization Through Polarizers

    Visual: A stylized light-wave icon (an arrow with oscillating sine-wave) approaches two crossed polarizers. Before measurement it’s semi-transparent and angled at, say, 30°.

    Interaction:

        Rotate Arrow: Drag to set the incoming polarization angle.

        Measure at Polarizer: Tap the polarizer to show the wave emerging either horizontal or vertical—randomly, with probability cos²(Δθ) and sin²(Δθ).

    Tech Tips:

        Represent the wave as a simple arrow + sine mesh, adjust its rotation.

        On measurement, sample probability and then animate the output direction snapping to 0° or 90°.
