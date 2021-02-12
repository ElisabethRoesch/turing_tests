# DEFINITIONS
# ref_turn_angle: angle calculated (atan2) betweem current motion vector
# and predefined ref_axis. Here the positive x-axis [1,0]. It is the same as the
# angle drawn directly from the VM distribution.
#

function bprw_2d(steps, bias, params)

    # Initialize arrays to store return variables
    bias_angles, angles, turn_angles, ref_turn_angles, coords, vectors, sources, persists, vecs = Float64[], Float64[], Float64[], Float64[], [], [], [], [], []
    x,y = zeros(steps), zeros(steps)
    timepoints = collect(1:steps)
    steps = zeros(steps)
    cell_ids = []

    # Declare ref axis: angles are drawn from VM distribution wrt to this positive x-axis by default in Julia
    # We therefore use this as the ref axis in our model
    ref_axis = [1.0,0.0]

    # Randomly select starting coordinate for cell. We draw a random angle from the VM distribution and compute x and y
    # using cos(angle) and sin(angle), respectively.
    p_angle = random_angle()
    coord_t0 = [cos(p_angle),sin(p_angle)]
    # coord_t0 = random_point()
    x[1], y[1] = coord_t0[1], coord_t0[2]

    # Compute the bias vector from the first point to the bias point
    first_bv = bias - coord_t0
    # Compute the angle between the ref axis vector and the vector from the
    # first coordinate to the bias point in the system
    # This is the angle to put in the VM distribution so that all subsequent
    # angles are drawn close to this angle if kappa is large
    beta = signed_angle_2d_atan(ref_axis, first_bv)

    # Level of persistence and bias in the model
    kp = params[2]
    kb = params[3]

    # Enter the random walk
    for i = 2:length(steps)

        # Compute a weighted bias/persistent angle which will describe the direction with which to update walker
        angle = rand(MixtureModel(VonMises, [(beta, kb), (p_angle, kp)], [params[1], (1-params[1])]))
        append!(angles, angle)
        # Sample a distance from a truncated normal distribution - can be any distribution of choice
        # So velocity is Gaussian distributed in the 2d model - need to research this
        step_length = rand(truncated(Normal(0.5, 1.0), 0.0, 1.0))

        # Update p_angle so it is always = to the angle of the previous step
        p_angle = angle

        # Update the position of the walker by moving it step_length units in the direction of angle
        x[i] = x[i-1] + step_length * cos(angle)
        y[i] = y[i-1] + step_length * sin(angle)

        # Compute the coordinate and motion vector
        coord = [x[i], y[i]]
        vector = [[x[i], y[i]] - [x[i-1], y[i-1]]]
        vec = ([x[i], y[i]] - [x[i-1], y[i-1]])
        # Compute turn_angle wrt ref_axis (pos x_axis)
        # Order very important
        # This should be exactly the same as angle
        ref_turn_angle = signed_angle_2d_atan(ref_axis, [x[i], y[i]] - [x[i-1], y[i-1]])
        # Compute the bias angle between current motion vector and bias_vector
        # Order very important
        bv = bias - [x[i-1], y[i-1]]
        # Compute angle between the bias vector and the ref axis (positive x-axis)
        # Use this as mu in the VM dist in the next iteration of the walk
        beta = signed_angle_2d_atan(ref_axis, bv)
        # Compute the bias angle
        bias_angle = signed_angle_2d_atan(bv, [x[i], y[i]] - [x[i-1], y[i-1]])

        # Save all values needed
        append!(ref_turn_angles, ref_turn_angle)
        append!(bias_angles, bias_angle)
        append!(coords, coord)
        append!(vectors, vector)
        append!(vecs, vec)
        append!(sources, beta)
        # This could be angle or ref_angle as they should be the same thing
        # Check that angle and ref_angle are the same values
        append!(persists, angle)
        end

        # Compute the turn angle = angle between consecutive motion vectors
        for i in 2:length(vectors)
        a = signed_angle_2d_atan(vectors[i-1], vectors[i])
        append!(turn_angles, a)
        end

        return x, y, angles, turn_angles, ref_turn_angles, bias_angles, coords, vectors, sources, persists, vecs
    end
