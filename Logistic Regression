# Logistic function (a.k.a., softmax function, sigmoid function)
def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def logit(x, beta):
    logit = np.dot(x, beta)
    return logit

beta = [0.1, -0.1]
y_hat = logistic(logit(x=x_train, beta=beta)) # Y = 1일 확률
print(y_hat[:10])
print(y_train[:10])

def negative_log_likelihood(x, y, beta, avg=False):
    y_hat = logistic(logit(x=x_train, beta=beta))
    log_likelihood = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    if not avg:
        return -log_likelihood
    else:
        return np.mean(-log_likelihood)
    
loss = negative_log_likelihood(x=x_train, y=y_train, beta=beta)
print(loss)


def logistic_regression(x, y, num_steps=50, learning_rate=0.01):
    initial_beta = np.asarray([10, 10])
    beta_list = []
    beta_list.append(initial_beta)
    beta = initial_beta
    
    for step in range(num_steps):
        y_hat = logistic(logit(x=x_train, beta=beta))
        gradient = np.dot(x.T, y_hat - y)
        new_beta = beta - learning_rate * gradient
        loss = negative_log_likelihood(x, y, beta, avg=True)
        
        # Print
        loss_tracking = '[%d step] loss %.4f' % (step, loss)
        beta_equation = '[%.2f %.2f] = [%.2f %.2f] - %.2f * [%.2f %.2f]' % (new_beta[0], new_beta[1], beta[0], beta[1], learning_rate, gradient[0], gradient[1])
        print(loss_tracking, '\t', beta_equation)
        
        beta_list.append(new_beta)
        beta = new_beta
        
    return beta, beta_list

beta, beta_list = logistic_regression(x=x_train, y=y_train, num_steps=100, learning_rate=0.01)

print(beta)
print(beta_list)


def vis_hyperplane(beta, style='k--'):

    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()
    m0, m1 = lim0[0], lim0[1]

    intercept0 = -(beta[0] * m0 + beta[-1])/beta[1]
    intercept1 = -(beta[0] * m1 + beta[-1])/beta[1]
    
    plt1, = plt.plot([m0, m1], [intercept0, intercept1], style)

    plt.gca().set_xlim(lim0)
    plt.gca().set_ylim(lim1)
        
    return plt1

# Training data + Hyperplane
plt.figure()
plt.scatter(x_train[(y_train==0),0], x_train[(y_train==0),1], c='blue', marker='o')
plt.scatter(x_train[(y_train==1),0], x_train[(y_train==1),1], c='red', marker='o')

initial_beta_plot = vis_hyperplane(beta_list[0], 'k--')
beta_plot_10 = vis_hyperplane(beta_list[9], 'b--')
beta_plot_15 = vis_hyperplane(beta_list[14], 'r--')
beta_plot = vis_hyperplane(beta, 'g--')

plt.legend([initial_beta_plot, beta_plot_10, beta_plot_15, beta_plot], ['Randomly initialized beta', 'Learned beta(10th updated)', 'Learned beta(15th updated)', 'Learned beta'])

# Test data + Hyperplane
plt.figure()
plt.scatter(x_test[(y_test==0),0], x_test[(y_test==0),1], c='blue', marker='+')
plt.scatter(x_test[(y_test==1),0], x_test[(y_test==1),1], c='red', marker='+')

initial_beta_plot = vis_hyperplane(beta_list[0], 'k--')
beta_plot = vis_hyperplane(beta, 'g--')

plt.legend([initial_beta_plot, beta_plot], ['Randomly initialized beta', 'Learned beta'])

# Training data + Test data
plt.figure()
plt.scatter(x_train[(y_train==0),0], x_train[(y_train==0),1], c='blue', marker='o', alpha=0.1)
plt.scatter(x_train[(y_train==1),0], x_train[(y_train==1),1], c='red', marker='o', alpha=0.1)
plt.scatter(x_test[(y_test==0),0], x_test[(y_test==0),1], c='blue', marker='+')
plt.scatter(x_test[(y_test==1),0], x_test[(y_test==1),1], c='red', marker='+')

initial_beta_plot = vis_hyperplane(beta_list[0], 'k--')
beta_plot = vis_hyperplane(beta, 'g--')

plt.legend([initial_beta_plot, beta_plot], ['Randomly initialized beta', 'Learned beta'])
