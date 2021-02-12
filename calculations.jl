#####################################################################
# FUNCTIONS
#####################################################################

# 1. COMPUTE SIGNED ANGLE BETWEEN 2 2D VECTORS USING ATAN2
function signed_angle_2d_atan(v1,v2)
    angle = atan(v1[1]*v2[2]-v1[2]*v2[1], v1[1]*v2[1]+v1[2]*v2[2])
    return angle
end

# 2. COMPUTE A TRANSITION MATRIX BASED ON AN ARRAY OF TURN ANGLES
function transition_matrix_ref(m,angles,bins)
    # Bin data
    binned_data = map(data -> hist(data, bins), angles)
    # Get the indexes of the filled bins (1-8)
    positions = Int[]
    for i in 1:length(angles)
        position = argmax(binned_data[i][1])
        append!(positions, position)
    end
    # Iterate over the indexes and find coordinates that correspond to a position on the heatmap
    # Add 1.0 each time a specific coordinate is selected
    coords = []
    for i = 2:length(positions)
        coord = (positions[i],positions[i-1])
        push!(coords,coord)
        m[coord[1],coord[2]] += 1
    end
    return m
end

# 3. COMPUTE K (STRENGTH PARAMETER IN VMF DISTRIBUTION) FROM A GIVEN P&B
function calculate_k(x)
    k = -5log(1-x)
    return k
end

# 4. GENERATE RANDOM POINT ON UNIT CIRCLE
function random_point()
    angle = rand(VonMises(0.0, 0.0000001))
    x = cos(angle)
    y = sin(angle)
    point = [x,y]
    return point
end

# 5. GENERATE RANDOM ANGLE ON UNIT CIRCLE BOUND BETWEEN -PI AND PI
function random_angle()
    angle = rand(VonMises(0.0, 0.0000001))
    return angle
end

# NORMALISING FUNCTION
normalise(x) = x/norm(x)
