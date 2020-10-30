function [N,E] = rentian_scaling_2d(A,XY,n,tol)
% RENTIAN_SCALING_2D    Rentian scaling for networks embedded in two dimensions.
%
% [N,E] = rentian_scaling_2d(A,XY,n,tol)
%
% Physical Rentian scaling (or more simply Rentian scaling) is a property
% of systems that are cost-efficiently embedded into physical space. It is
% what is called a "topo-physical" property because it combines information
% regarding the topological organization of the graph with information
% about the physical placement of connections. Rentian scaling is present
% in very large scale integrated circuits, the C. elegans neuronal network,
% and morphometric and diffusion-based graphs of human anatomical networks.
% Rentian scaling is determined by partitioning the system into cubes,
% counting the number of nodes inside of each cube (N), and the number of
% edges traversing the boundary of each cube (E). If the system displays
% Rentian scaling, these two variables N and E will scale with one another
% in loglog space. The Rent's exponent is given by the slope of log10(E)
% vs. log10(N), and can be reported alone or can be compared to the
% theoretical minimum Rent's exponent to determine how cost efficiently the
% network has been embedded into physical space. Note: if a system displays
% Rentian scaling, it does not automatically mean that the system is
% cost-efficiently embedded (although it does suggest that). Validation
% occurs when comparing to the theoretical minimum Rent's exponent for that
% system.
%
% INPUTS:
%
% A:              MxM adjacency matrix.
%                 Must be unweighted, binary, and symmetric.
% XY:             Matrix of node placement coordinates.
%                 Must be in the form of an Mx2 matrix [x y], where M is the
%                 number of nodes and x and y are column vectors of node
%                 coordinates.
% n:              Number of partitions to compute. Each partition is a data
%                 point. You want a large enough number to adequately
%                 estimate the Rent's exponent.
% tol:            This should be a small value (for example 1e-6).
%                 In order to mitigate the effects of boundary conditions due
%                 to the finite size of the network, we only allow partitions
%                 that are contained within the boundary of the network. This
%                 is achieved by first computing the volume of the convex
%                 hull of the node coordinates (V). We then ensure that the
%                 volume of the convex hull computed on the original node
%                 coordinates plus the coordinates of the randomly generated
%                 partition (Vnew) is within a given tolerance of the
%                 original (i.e. check abs(V - Vnew) < tol). Thus tol, should
%                 be a small value in order to make sure the partitions are
%                 contained largely within the boundary of the network, and
%                 thus the number of nodes and edges within the box are not
%                 skewed by finite size effects.
%
% OUTPUTS:
%
% N:              nx1 vector of the number of nodes in each of the n partitions.
% E:              nx1 vector of the number of edges crossing the boundary of
%                 each partition.
%
% Subsequent Analysis:
%
%     Rentian scaling plots are created by: figure; loglog(E,N,'*');
%
%     To determine the Rent's exponent, p, we need to determine
%     the slope of E vs. N in loglog space, which is the Rent's
%     exponent. There are many ways of doing this with more or less
%     statistical rigor. Robustfit in MATLAB is one such option:
%
%         [b,stats] = robustfit(log10(N),log10(E))
%
%     Then the Rent's exponent is b(1,2) and the standard error of the
%     estimation is given by stats.se(1,2).
%
% Note: n=5000 was used in Bassett et al. 2010 in PLoS CB.
%
% Reference:
% Danielle S. Bassett, Daniel L. Greenfield, Andreas Meyer-Lindenberg,
% Daniel R. Weinberger, Simon W. Moore, Edward T. Bullmore. Efficient
% physical embedding of topologically complex information processing
% networks in brains and computer circuits. PLoS Comput Biol, 2010,
% 6(4):e1000748.
%
% Modification History:
%
%     2010:     Original (Dani Bassett)
%     Dec 2016: Updated code so that both partition centers and partition
%               sizes are chosen at random. Also added in a constraint on
%               partition placement that prevents boxes from being located
%               outside the edges of the network. This helps prevent skewed
%               results due to boundary effects arising from the finite size
%               of the network. (Lia Papadopoulos)

% determine the number of nodes in the system
M = numel(XY(:,1));

% rescale coordinates so that they are all greater than unity
XYn = XY - repmat(min(XY)-1,M,1);

% compute the area of convex hull (i.e. are of the boundary) of the network
[~,V] = convhull(XYn(:,1),XYn(:,2));

% min and max network coordinates
xmin = min(XYn(:,1));
xmax = max(XYn(:,1));
ymin = min(XYn(:,2));
ymax = max(XYn(:,2));

% initialize vectors of number of nodes in box and number of edges crossing
% box

N = zeros(n,1);
E = zeros(n,1);

% create partitions, and count the number of nodes inside the partition (N)
% and the number of edges traversing the boundary of the partition (E)

nPartitions = 0;

while nPartitions<(n+1)
    
    % variable to check if partition center is within network boundary
    % OK if inside == 1
    inside = 0;
    
    while inside == 0
        
        % pick a random (x,y) coordinate to be the center of the box
        randx = xmin+(xmax-xmin)*rand(1);
        randy = ymin+(ymax-ymin)*rand(1);
        
        % make sure the point is inside the convex hull of the network
        newCoords = [XYn; [randx randy]];
        [~,Vnew] = convhull(newCoords(:,1),newCoords(:,2));
        
        % if the old convex hull area and new convex hull area are equal
        % then the box center must be inside the network boundary.
        
        if isequal(V,Vnew)==0
            inside = 0;
        else
            inside = 1;
        end
        
    end
    
    % determine the approximate maximum distance the box can extend, given
    % the center point and the  bounds of the network
    deltaY = min(abs(ymax-randy),abs(ymin-randy));
    deltaX = min(abs(xmax-randx),abs(xmin-randx));
    deltaLmin = min(deltaY,deltaX);
    
    % variable to check if partition is within network boundary
    % OK if inside == 1
    inside = 0;
    
    while inside == 0
        
        % pick a random (side length)/2 that is between 0 and the
        % max possible
        deltaL = deltaLmin*rand(1);
        
        % (x,y) coordinates for corners of box
        boxCoords = [randx - deltaL randy - deltaL; ...
            randx - deltaL randy + deltaL; ...
            randx + deltaL randy - deltaL; ...
            randx + deltaL randy + deltaL];
        
        % check if all corners of box are inside the convex hull of the
        % network
        newCoords = [XYn; boxCoords];
        [~,Vnew] = convhull(newCoords(:,1),newCoords(:,2));
        
        % make sure the new convex hull that includes the partition corners
        % is within a certain tolerance of the original convex hull area.
        
        if abs(V-Vnew)>tol
            inside = 0;
        else
            inside = 1;
        end
    end
    
    % Find nodes inside the box, edges crossing the boundary
    
    L = find(XYn(:,1)>(randx-deltaL) & XYn(:,1)<(randx+deltaL) ...
        & XYn(:,2)>(randy-deltaL) & XYn(:,2)<(randy+deltaL));
    
    if ~isempty(L) == 1
        nPartitions = nPartitions+1;
        % count edges crossing the boundary of the box
        E(nPartitions,1) = sum(sum(A(L,setdiff(1:M,L))));
        % count nodes inside of the box
        N(nPartitions,1) = numel(L);
        
    end
    
end

return;