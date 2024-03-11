export const postgresConfig = {
   database: String(process.env.DB_NAME),
   keepAlive: true,
   max: 100,
   password: String(process.env.DB_PASSWORD),
   user: String(process.env.DB_USER),
}